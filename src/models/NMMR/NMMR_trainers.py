import os.path as op
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.ate.data_class import (PVTestDataSetTorch, PVTrainDataSetTorch,
                                     RHCTestDataSet)
from src.models.NMMR.kernel_utils import (calculate_kernel_matrix,
                                          calculate_kernel_matrix_batched,
                                          rbf_kernel)
from src.models.NMMR.NMMR_loss import NMMR_loss, NMMR_loss_batched
from src.models.NMMR.NMMR_model import cnn_mono_for_dsprite
from src.models.shared.shared_models import (cnn_for_dsprite, mlp_for_dsprite,
                                             mlp_general)
from src.utils.select_device import select_device


class NMMR_Trainer_RHCExperiment:
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                    analysis_config: Dict[str, Any], random_seed: int,
                    dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):
        self.data_config = data_configs
        self.train_params = train_params
        self.analysis_config = analysis_config
        self.device_name = select_device(preferred_device)
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']
        self.l2_lambda = train_params['l2_lambda']
        self.log_metrics = analysis_config['log_metrics']

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):
        return calculate_kernel_matrix(kernel_inputs)

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0):
        
        if self.device_name != "cpu":
            train_t = train_t.to_gpu(self.device_name)
            val_t = val_t.to_gpu(self.device_name)

        n_sample = len(train_t.treatment)
        dim_A = train_t.treatment.shape[1]
        dim_W = train_t.outcome_proxy.shape[1]
        dim_Z = train_t.treatment_proxy.shape[1]
        dim_X = train_t.backdoor.shape[1]
        dim_y = train_t.outcome.shape[1]
        
        model = MLP_for_NMMR(input_dim = (dim_A + dim_W + dim_X), train_params=self.train_params)

        if self.device_name != "cpu":
            model.to(self.device_name)

        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(n_sample)

            for i in range(0, n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A = train_t.treatment[indices]
                batch_W = train_t.outcome_proxy[indices]
                batch_Z = train_t.treatment_proxy[indices]
                batch_X = train_t.backdoor[indices]
                batch_y = train_t.outcome[indices]

                optimizer.zero_grad()
                batch_inputs = torch.cat((batch_A, batch_W, batch_X), dim=1)
                pred_y = model(batch_inputs)

                # TODO: check that kernel matrix isn't too dominated by X's (vs. A and Z)
                kernel_inputs_train = torch.cat((batch_A, batch_Z, batch_X), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train,
                                                    self.loss_name, self.l2_lambda)[0]
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(torch.cat((train_t.treatment, train_t.outcome_proxy, train_t.backdoor), dim=1))
                    preds_val = model(torch.cat((val_t.treatment, val_t.outcome_proxy, val_t.backdoor), dim=1))

                    # compute the full kernel matrix
                    kernel_inputs_train = torch.cat((train_t.treatment, train_t.treatment_proxy, train_t.backdoor), dim=1)
                    kernel_inputs_val = torch.cat((val_t.treatment, val_t.treatment_proxy, val_t.backdoor), dim=1)
                    kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = self.mse_loss(preds_train, train_t.outcome)
                    mse_val = self.mse_loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss(preds_train, train_t.outcome, kernel_matrix_train, self.loss_name,
                                                    self.l2_lambda)
                    causal_loss_val = NMMR_loss(preds_val, val_t.outcome, kernel_matrix_val, self.loss_name,
                                                    self.l2_lambda)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model

    @staticmethod
    def predict(model, test_data_t: RHCTestDataSet):
        # Create a 3-dim array with shape [n_treatments, n_samples, len(A) + len(W) + len(X)]
        # The first axis contains the two values of do(A): 0 and 1
        # The last axis contains W, X, needed for the model's forward pass
        n_treatments = len(test_data_t.treatment)
        n_samples = len(test_data_t.outcome_proxy)
        tempA = test_data_t.treatment.unsqueeze(-1).expand(-1, n_samples, -1)
        tempW = test_data_t.outcome_proxy.unsqueeze(0).expand(n_treatments, -1, -1)
        tempX = test_data_t.backdoor.unsqueeze(0).expand(n_treatments, -1, -1)
        model_inputs_test = torch.dstack((tempA, tempW, tempX))

        # Compute model's predicted E[Y | do(A)] = E_{w, x}[h(a, w, x)] for A in [0, 1]
        # Note: the mean is taken over the n_samples axis, so we obtain 2 avg. pot. outcomes; their diff is the ATE
        with torch.no_grad():
            pred = torch.mean(model(model_inputs_test), dim=1)

        return pred.flatten().cpu().detach().numpy()

    @staticmethod
    def norm(model, val_data_t: PVTrainDataSetTorch):
        n = len(val_data_t.treatment)
        val_A = val_data_t.treatment
        val_W = val_data_t.outcome_proxy
        val_X = val_data_t.backdoor
        model_inputs_val = torch.cat((val_A, val_W, val_X), dim=1)

        with torch.no_grad():
            model_output = model(model_inputs_val)
            norm = torch.sqrt( ( model_output.T @ model_output ) / n )

        return norm[0,0].cpu().detach().numpy()


class NMMR_Trainer_DemandExperiment(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                    analysis_config: Dict[str, Any], random_seed: int,
                    dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):
        self.data_config = data_configs
        self.train_params = train_params
        self.analysis_config = analysis_config
        self.device_name = select_device(preferred_device)
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']
        self.l2_lambda = train_params['l2_lambda']
        self.log_metrics = analysis_config['log_metrics']

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):
        return calculate_kernel_matrix(kernel_inputs)

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0) -> MLP_for_NMMR:
       
        if self.device_name != "cpu":
            train_t = train_t.to_gpu(self.device_name)
            val_t = val_t.to_gpu(self.device_name)
  
        n_sample = len(train_t.treatment)
        dim_A = train_t.treatment.shape[1]
        dim_W = train_t.outcome_proxy.shape[1]
        dim_Z = train_t.treatment_proxy.shape[1]
        dim_X = train_t.backdoor.shape[1]
        dim_y = train_t.outcome.shape[1]

        model = MLP_for_NMMR(input_dim = (dim_A + dim_W + dim_X), train_params=self.train_params)

        if self.device_name != "cpu":
            model.to(self.device_name)

        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(n_sample)

            for i in range(0, n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A = train_t.treatment[indices]
                batch_W = train_t.outcome_proxy[indices]
                batch_Z = train_t.treatment_proxy[indices]
                batch_X = train_t.backdoor[indices]
                batch_y = train_t.outcome[indices]

                optimizer.zero_grad()
                batch_inputs = torch.cat((batch_A, batch_W, batch_X), dim=1)
                pred_y = model(batch_inputs)
                kernel_inputs_train = torch.cat((batch_A, batch_Z, batch_X), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train,
                                                    self.loss_name, self.l2_lambda)[0]
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(torch.cat((train_t.treatment, train_t.outcome_proxy, train_t.backdoor), dim=1))
                    preds_val = model(torch.cat((val_t.treatment, val_t.outcome_proxy, val_t.backdoor), dim=1))

                    # compute the full kernel matrix
                    kernel_inputs_train = torch.cat((train_t.treatment, train_t.treatment_proxy, train_t.backdoor), dim=1)
                    kernel_inputs_val = torch.cat((val_t.treatment, val_t.treatment_proxy, val_t.backdoor), dim=1)
                    kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = self.mse_loss(preds_train, train_t.outcome)
                    mse_val = self.mse_loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss(preds_train, train_t.outcome, kernel_matrix_train, self.loss_name,
                                                    self.l2_lambda)
                    causal_loss_val = NMMR_loss(preds_val, val_t.outcome, kernel_matrix_val, self.loss_name,
                                                    self.l2_lambda)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch):
        # Create a 3-dim array with shape [n_treatments, n_samples, len(A) + len(W) + len(X)]
        # The first axis contains the test values for do(A) chosen by Xu et al.
        # The last axis contains random draws for W (and X) needed for the model's forward pass.
        n_treatments = len(test_data_t.treatment)
        n_samples = len(val_data_t.outcome_proxy)
        tempA = test_data_t.treatment.unsqueeze(-1).expand(-1, n_samples, -1)
        tempW = val_data_t.outcome_proxy.unsqueeze(0).expand(n_treatments, -1, -1)
        tempX = val_data_t.backdoor.unsqueeze(0).expand(n_treatments, -1, -1)
        model_inputs_test = torch.dstack((tempA, tempW, tempX))

        # Compute model's predicted E[Y | do(A)] = E_{w, x}[h(a, w, x)]
        # Note: the mean is taken over the n_samples axis, so we obtain {n_treatments} number of expected values
        with torch.no_grad():
            pred = torch.mean(model(model_inputs_test), dim=1)

        return pred.flatten().cpu().detach().numpy()

    @staticmethod
    def norm(model, val_data_t: PVTrainDataSetTorch):
        n = len(val_data_t.treatment)
        val_A = val_data_t.treatment
        val_W = val_data_t.outcome_proxy
        val_X = val_data_t.backdoor
        model_inputs_val = torch.cat((val_A, val_W, val_X), dim=1)

class NMMR_Trainer_dSpriteExperiment_mod(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], random_seed: int,
                 dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):

        self.data_config = data_configs
        self.train_params = train_params
        self.analysis_config = analysis_config
        self.device_name = select_device(preferred_device)
        self.n_sample = self.data_config['n_sample']
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.val_batch_size = train_params['val_batch_size']
        self.kernel_batch_size = train_params['kernel_batch_size']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']
        self.cnn_type = train_params['cnn_type']
        self.l2_lambda = train_params['l2_lambda']
        self.log_metrics = analysis_config['log_metrics']
        self.A_scale = 0.05  # TODO: tune this value? Looked pretty good as is

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):

        return calculate_kernel_matrix(kernel_inputs)
        
    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0) -> Tuple[cnn_for_dsprite, cnn_for_dsprite, mlp_for_dsprite]:

        n_sample = len(train_t.treatment)
        dim_A = train_t.treatment.shape[1]
        dim_W = train_t.outcome_proxy.shape[1]
        dim_Z = train_t.treatment_proxy.shape[1]
        dim_X = train_t.backdoor.shape[1]
        dim_y = train_t.outcome.shape[1]
        dim_CNN = 128                           # Output dimension of CNNs

        # CNNs to derive hidden A and W parameters from images and main MLP
        model_a = cnn_for_dsprite(train_params=self.train_params)
        model_w = cnn_for_dsprite(train_params=self.train_params)
        model = mlp_for_dsprite(input_dim = (dim_CNN + dim_CNN + dim_X),train_params=self.train_params)

        # Move data and NNs to GPU
        if self.device_name != "cpu":
            train_t = train_t.to_gpu(self.device_name)
            val_t = val_t.to_gpu(self.device_name)
            model_a.to(self.device_name)
            model_w.to(self.device_name)
            model.to(self.device_name)

        # Optimizers for each NN
        optimizer_a = optim.Adam(list(model_a.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer_w = optim.Adam(list(model_w.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A = train_t.treatment[indices]
                batch_W = train_t.outcome_proxy[indices]
                batch_Z = train_t.treatment_proxy[indices]
                batch_X = train_t.backdoor[indices]
                batch_y = train_t.outcome[indices]

                pred_A = model_a(batch_A.reshape(-1, 1, 64, 64))
                pred_W = model_w(batch_W.reshape(-1, 1, 64, 64))

                batch_inputs = torch.cat((pred_A, pred_W, batch_X), dim=1)
                pred_y = model(batch_inputs)
                kernel_inputs_train = torch.cat((self.A_scale * batch_A,
                    batch_Z, batch_X), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train, self.loss_name, self.l2_lambda)
                
                optimizer.zero_grad()
                optimizer_a.zero_grad()
                optimizer_w.zero_grad()
                causal_loss_train.backward()
                optimizer.step()
                optimizer_a.step()
                optimizer_w.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    pred_A = model_a(train_t.treatment.reshape(-1, 1, 64, 64))
                    pred_W = model_w(train_t.outcome_proxy.reshape(-1, 1, 64, 64))
                    batch_inputs = torch.cat((pred_A, pred_W, train_t.backdoor), dim=1)
                    preds_train = model(batch_inputs)
                    pred_A = model_a(val_t.treatment.reshape(-1, 1, 64, 64))
                    pred_W = model_w(val_t.outcome_proxy.reshape(-1, 1, 64, 64))
                    batch_inputs = torch.cat((pred_A, pred_W, train_t.backdoor), dim=1)
                    preds_val = model(batch_inputs)
                    kernel_inputs_train = torch.cat((self.A_scale * train_t.treatment,
                                        train_t.treatment_proxy, train_t.backdoor), dim=1)
                    kernel_inputs_val = torch.cat((self.A_scale * val_t.treatment,
                                        val_t.treatment_proxy, val_t.backdoor), dim=1)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = self.mse_loss(preds_train, train_t.outcome)
                    mse_val = self.mse_loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss_batched(preds_train, train_t.outcome, kernel_inputs_train, rbf_kernel,
                                                        self.kernel_batch_size, self.loss_name, self.l2_lambda)
                    causal_loss_val = NMMR_loss_batched(preds_val, val_t.outcome, kernel_inputs_val, rbf_kernel,
                                                        self.kernel_batch_size, self.loss_name, self.l2_lambda)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model_a, model_w, model

    @staticmethod
    def predict(model_a, model_w, model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch, batch_size=None):

        intervention_array_len = test_data_t.treatment.shape[0]
        num_W_test = val_data_t.outcome_proxy.shape[0]

        mean = torch.nn.AvgPool1d(kernel_size=num_W_test, stride=num_W_test)
        with torch.no_grad():
            if batch_size is None:
                # create n_sample copies of each test image (A), and 588 copies of each proxy image (W)
                # reshape test and proxy image to 1 x 64 x 64 (so that the model's conv2d layer is happy)
                test_A = test_data_t.treatment.repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                test_W = val_data_t.outcome_proxy.repeat(intervention_array_len, 1).reshape(-1, 1, 64, 64)
                test_X = val_data_t.backdoor.repeat(intervention_array_len, 1)
                pred_A = model_a(test_A)
                pred_W = model_w(test_W)
                batch_inputs = torch.cat((pred_A, pred_W, test_X), dim=1)
                E_w_haw = mean(model(batch_inputs).unsqueeze(-1).T)
            else:
                # the number of A's to evaluate each batch
                a_step = max(1, batch_size // num_W_test)
                E_w_haw = torch.zeros([1, 1, intervention_array_len])
                for a_idx in range(0, intervention_array_len, a_step):
                    temp_A = test_data_t.treatment[a_idx:(a_idx + a_step)].repeat_interleave(num_W_test, dim=0).reshape(
                        -1, 1, 64, 64)
                    temp_W = val_data_t.outcome_proxy.repeat(a_step, 1).reshape(-1, 1, 64, 64)
                    temp_X = val_data_t.backdoor.repeat(a_step, 1)
                    # in this case, we're only predicting for a single A, so we have a ton of W's
                    # therefore, we'll batch this step as well
                    if a_step == 1:
                        model_preds = torch.zeros((temp_A.shape[0]))
                        for temp_idx in range(0, temp_A.shape[0], batch_size):
                            pred_A = model_a(temp_A[temp_idx:temp_idx + batch_size])
                            pred_W = model_w(temp_W[temp_idx:temp_idx + batch_size])
                            batch_inputs = torch.cat((pred_A, pred_W, temp_X), dim=1)
                            model_preds[temp_idx:(temp_idx + batch_size)] = model(
                                batch_inputs).squeeze()
                        E_w_haw[0, 0, a_idx] = torch.mean(model_preds)
                    else:
                        pred_A = model_a(temp_A)
                        pred_W = model_w(temp_W)
                        batch_inputs = torch.cat((pred_A, pred_W, temp_X), dim=1)
                        temp_E_w_haw = mean(model(batch_inputs).unsqueeze(-1).T)
                        E_w_haw[0, 0, a_idx:(a_idx + a_step)] = temp_E_w_haw[0, 0]

        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values

        return E_w_haw.T.squeeze(1).cpu().detach().numpy()

    @staticmethod
    def norm(model_a, model_w, model, val_data_t: PVTrainDataSetTorch, batch_size=None):

        n = len(val_data_t.treatment)
        with torch.no_grad():
            if batch_size is None:
                val_A = val_data_t.treatment
                val_W = val_data_t.outcome_proxy
                val_X = val_data_t.backdoor
                pred_A = model_a(val_A)
                pred_W = model_w(val_W)
                batch_inputs = torch.cat((pred_A, pred_W, val_X), dim=1)
                model_output = model(batch_inputs)
                norm = torch.sqrt( ( model_output.T @ model_output ) / n )
            else:
                norm2 = 0
                for temp_idx in range(0, n, batch_size):
                    pred_A = model_a(val_A[temp_idx:temp_idx + batch_size])
                    pred_W = model_w(val_W[temp_idx:temp_idx + batch_size])
                    temp_X = val_X[temp_idx:temp_idx + batch_size]
                    batch_inputs = torch.cat((pred_A, pred_W, temp_X), dim=1)
                    model_output = model(batch_inputs)
                    temp_norm2 = ( model_output.T @ model_output )
                    norm2 += temp_norm2

                norm = torch.sqrt( norm2 / n )

            return norm[0,0].cpu().detach().numpy()

class NMMR_Trainer_dSpriteExperiment_mono(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                    analysis_config: Dict[str, Any], random_seed: int,
                    dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):
        self.data_config = data_configs
        self.train_params = train_params
        self.analysis_config = analysis_config
        self.device_name = select_device(preferred_device)
        self.n_sample = self.data_config['n_sample']
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.val_batch_size = train_params['val_batch_size']
        self.kernel_batch_size = train_params['kernel_batch_size']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']
        self.cnn_type = train_params['cnn_type']
        self.l2_lambda = train_params['l2_lambda']
        self.log_metrics = analysis_config['log_metrics']
        self.A_scale = 0.05  # TODO: tune this value? Looked pretty good as is

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):

        return calculate_kernel_matrix(kernel_inputs)

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0) -> cnn_mono_for_dsprite:

        # inputs consist of (A, W) tuples
        model = cnn_mono_for_dsprite(train_params=self.train_params)

        if self.device_name != "cpu":
            train_t = train_t.to_gpu(self.device_name)
            val_t = val_t.to_gpu(self.device_name)
            model.to(self.device_name)

        # weight_decay implements L2 penalty on weights
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A, batch_W, batch_y = ( train_t.treatment[indices], train_t.outcome_proxy[indices],
                                                train_t.outcome[indices] )

                optimizer.zero_grad()
                pred_y = model(batch_A.reshape(-1, 1, 64, 64), batch_W.reshape(-1, 1, 64, 64))
                kernel_inputs_train = torch.cat(
                    (self.A_scale * train_t.treatment[indices], train_t.treatment_proxy[indices]),
                    dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train,
                                                    self.loss_name, self.l2_lambda)[0]
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(train_t.treatment.reshape(-1, 1, 64, 64),
                                        train_t.outcome_proxy.reshape(-1, 1, 64, 64))
                    preds_val = model(val_t.treatment.reshape(-1, 1, 64, 64),
                                        val_t.outcome_proxy.reshape(-1, 1, 64, 64))
                    kernel_inputs_train = torch.cat((self.A_scale * train_t.treatment, train_t.treatment_proxy), dim=1)
                    kernel_inputs_val = torch.cat((self.A_scale * val_t.treatment, val_t.treatment_proxy), dim=1)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = self.mse_loss(preds_train, train_t.outcome)
                    mse_val = self.mse_loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss_batched(preds_train, train_t.outcome, kernel_inputs_train, rbf_kernel,
                                                        self.kernel_batch_size, self.loss_name, self.l2_lambda)
                    causal_loss_val = NMMR_loss_batched(preds_val, val_t.outcome, kernel_inputs_val, rbf_kernel,
                                                        self.kernel_batch_size, self.loss_name, self.l2_lambda)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch, batch_size=None):

        intervention_array_len = test_data_t.treatment.shape[0]
        num_W_test = val_data_t.outcome_proxy.shape[0]

        mean = torch.nn.AvgPool1d(kernel_size=num_W_test, stride=num_W_test)
        with torch.no_grad():
            if batch_size is None:
                # create n_sample copies of each test image (A), and 588 copies of each proxy image (W)
                # reshape test and proxy image to 1 x 64 x 64 (so that the model's conv2d layer is happy)
                test_A = test_data_t.treatment.repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                test_W = val_data_t.outcome_proxy.repeat(intervention_array_len, 1).reshape(-1, 1, 64, 64)
                E_w_haw = mean(model(test_A, test_W).unsqueeze(-1).T)
            else:
                # the number of A's to evaluate each batch
                a_step = max(1, batch_size // num_W_test)
                E_w_haw = torch.zeros([1, 1, intervention_array_len])
                for a_idx in range(0, intervention_array_len, a_step):
                    temp_A = test_data_t.treatment[a_idx:(a_idx + a_step)].repeat_interleave(num_W_test, dim=0).reshape(
                        -1, 1, 64, 64)
                    temp_W = val_data_t.outcome_proxy.repeat(a_step, 1).reshape(-1, 1, 64, 64)
                    # in this case, we're only predicting for a single A, so we have a ton of W's
                    # therefore, we'll batch this step as well
                    if a_step == 1:
                        model_preds = torch.zeros((temp_A.shape[0]))
                        for temp_idx in range(0, temp_A.shape[0], batch_size):
                            model_preds[temp_idx:(temp_idx + batch_size)] = model(
                                temp_A[temp_idx:temp_idx + batch_size],
                                temp_W[temp_idx:temp_idx + batch_size]).squeeze()
                        E_w_haw[0, 0, a_idx] = torch.mean(model_preds)
                    else:
                        temp_E_w_haw = mean(model(temp_A, temp_W).unsqueeze(-1).T)
                        E_w_haw[0, 0, a_idx:(a_idx + a_step)] = temp_E_w_haw[0, 0]

        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values

        return E_w_haw.T.squeeze(1).cpu().detach().numpy()

    @staticmethod
    def norm(model, val_data_t: PVTrainDataSetTorch, batch_size=None):

        n = len(val_data_t.treatment)
        with torch.no_grad():
            if batch_size is None:
                val_A = val_data_t.treatment
                val_W = val_data_t.outcome_proxy
                val_X = val_data_t.backdoor
                batch_inputs = torch.cat((val_A, val_W, val_X), dim=1)
                model_output = model(batch_inputs)
                norm = torch.sqrt( ( model_output.T @ model_output ) / n )
            else:
                norm2 = 0
                for temp_idx in range(0, n, batch_size):
                    temp_A = val_A[temp_idx:temp_idx + batch_size]
                    temp_W = val_W[temp_idx:temp_idx + batch_size]
                    temp_X = val_X[temp_idx:temp_idx + batch_size]
                    batch_inputs = torch.cat((temp_A, temp_W, temp_X), dim=1)
                    model_output = model(batch_inputs)
                    temp_norm2 = ( model_output.T @ model_output )
                    norm2 += temp_norm2

                norm = torch.sqrt( norm2 / n )

            return norm[0,0].cpu().detach().numpy()
