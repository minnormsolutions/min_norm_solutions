import os.path as op
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.ate.data_class import PVTestDataSetTorch, PVTrainDataSetTorch
from src.models.naive_neural_net.naive_nn_model import (
    cnn_AWZY_mono_for_dsprite, cnn_AY_mono_for_dsprite,)
from src.models.shared.shared_models import (cnn_for_dsprite, mlp_for_dsprite,
                                             mlp_general)
from src.utils.make_AWZ_test import make_AWZ_test
from src.utils.select_device import select_device


class Naive_NN_Trainer_DemandExperiment(object):
    def __init__(self, data_config: Dict[str, Any], train_params: Dict[str, Any],
                    analysis_config: Dict[str, Any], random_seed: int,
                    dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):
        self.data_config = data_config
        self.train_params = train_params
        self.analysis_config = analysis_config
        self.device_name = select_device(preferred_device)
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.n_sample = data_config['n_sample']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.model_name = train_params['name']
        self.log_metrics = analysis_config['log_metrics']

        if self.log_metrics and (dump_folder is not None):
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.train_losses = []
            self.val_losses = []

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0) -> mlp_general:
        n_sample = len(train_t.treatment)
        dim_A = train_t.treatment.shape[1]
        dim_W = train_t.outcome_proxy.shape[1]
        dim_Z = train_t.treatment_proxy.shape[1]
        dim_X = train_t.backdoor.shape[1]
        dim_y = train_t.outcome.shape[1]

        if self.model_name == "naive_neural_net_AY":
            # inputs consist of only A
            model = mlp_general(input_dim = (dim_A + dim_X),
                                    train_params=self.train_params)
        elif self.model_name == "naive_neural_net_AWZY":
            # inputs consist of A, W, and Z (and Z is 2-dimensional)
            model = mlp_general(input_dim = (dim_A + dim_W + dim_Z + dim_X),
                                    train_params=self.train_params)
        else:
            raise ValueError(f"name {self.model_name} is not known")

        if self.device_name != "cpu":
            train_t = train_t.to_gpu(self.device_name)
            val_t = val_t.to_gpu(self.device_name)
            model.to(self.device_name)

        # weight_decay implements L2 penalty on weights
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss = nn.MSELoss()

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A = train_t.treatment[indices]
                batch_W = train_t.outcome_proxy[indices]
                batch_Z = train_t.treatment_proxy[indices]
                batch_X = train_t.backdoor[indices]
                batch_y = train_t.outcome[indices]

                if self.model_name == "naive_neural_net_AY":
                    batch_inputs = torch.cat((batch_A, batch_X), dim=1)

                if self.model_name == "naive_neural_net_AWZY":
                    batch_inputs = torch.cat((batch_A, batch_W, batch_Z, batch_X), dim=1)

                # training loop
                optimizer.zero_grad()
                pred_y = model(batch_inputs)
                output = loss(pred_y, batch_y)
                output.backward()
                optimizer.step()

            if self.log_metrics:
                with torch.no_grad():
                    if self.model_name == "naive_neural_net_AY":
                        preds_train = model(train_t.treatment)
                        preds_val = model(val_t.treatment)
                    elif self.model_name == "naive_neural_net_AWZY":
                        preds_train = model(torch.cat((train_t.treatment, train_t.outcome_proxy, train_t.treatment_proxy), dim=1))
                        preds_val = model(torch.cat((val_t.treatment, val_t.outcome_proxy, val_t.treatment_proxy), dim=1))

                        # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = loss(preds_train, train_t.outcome)
                    mse_val = loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)
                    self.train_losses.append(mse_train)
                    self.val_losses.append(mse_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch):
        if model.train_params['name'] == "naive_neural_net_AY":
            pred = model(test_data_t.treatment)

        elif model.train_params['name'] == "naive_neural_net_AWZY":
            AWZ_test = make_AWZ_test(test_data_t, val_data_t)
            pred = torch.mean(model(AWZ_test), dim=1)

        return pred.flatten().cpu().detach().numpy()

    @staticmethod
    def norm(model, val_data_t: PVTrainDataSetTorch):
        n = len(val_data_t.treatment)
        val_A = val_data_t.treatment
        val_W = val_data_t.outcome_proxy
        val_X = val_data_t.backdoor
        val_Z = val_data_t.treatment_proxy

        if model.train_params['name'] == "naive_neural_net_AY":
            model_inputs_val = torch.cat((val_A, val_X), dim=1)
        elif model.train_params['name'] == "naive_neural_net_AWZY":
            model_inputs_val = torch.cat((val_A, val_W, val_Z, val_X), dim=1)

        with torch.no_grad():
            model_output = model(model_inputs_val)
            norm = torch.sqrt( ( model_output.T @ model_output ) / n )

        return norm[0,0].cpu().detach().numpy()

class Naive_NN_Trainer_dSpriteExperiment(object):
    def __init__(self, data_config: Dict[str, Any], train_params: Dict[str, Any],
                    analysis_config: Dict[str, Any], random_seed: int,
                    dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):
        self.data_config = data_config
        self.train_params = train_params
        self.analysis_config = analysis_config
        self.device_name = select_device(preferred_device)
        self.data_name = data_config.get("name", None)
        self.n_sample = data_config['n_sample']
        self.model_name = train_params['name']
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.log_metrics = analysis_config['log_metrics']

        if self.log_metrics and (dump_folder is not None):
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.train_losses = []
            self.val_losses = []

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0):
        n_sample = len(train_t.treatment)
        dim_A = train_t.treatment.shape[1]
        dim_W = train_t.outcome_proxy.shape[1]
        dim_Z = train_t.treatment_proxy.shape[1]
        dim_X = train_t.backdoor.shape[1]
        dim_y = train_t.outcome.shape[1]
        dim_CNN = 128                           # Output dimension of CNNs
        
        if self.model_name == "naive_neural_net_AY":
            # inputs consist of only A
            model_a = cnn_for_dsprite(train_params=self.train_params)
            model = mlp_for_dsprite(input_dim = (dim_CNN + dim_X), train_params=self.train_params)
        elif self.model_name == "naive_neural_net_AWZY":
            # inputs consist of A, W, and Z (and Z is 3-dimensional)
            model_a = cnn_for_dsprite(train_params=self.train_params)
            model_w = cnn_for_dsprite(train_params=self.train_params)
            model = mlp_for_dsprite(input_dim = (dim_CNN + dim_CNN + dim_Z + dim_X),
                                        train_params=self.train_params)
        else:
            raise ValueError(f"name {self.model_name} is not known")

        # Move data and NNs to GPU
        if self.device_name != "cpu":
            train_t = train_t.to_gpu(self.device_name)
            val_t = val_t.to_gpu(self.device_name)
            model_a.to(self.device_name)
            if self.model_name == "naive_neural_net_AWZY":
                model_w.to(self.device_name)
            model.to(self.device_name)

        # Optimizers for each NN
        optimizer_a = optim.Adam(list(model_a.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.model_name == "naive_neural_net_AWZY":
            optimizer_w = optim.Adam(list(model_w.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        loss = nn.MSELoss()

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):

                indices = permutation[i:i + self.batch_size]
                batch_A = train_t.treatment[indices]
                batch_W = train_t.outcome_proxy[indices]
                batch_Z = train_t.treatment_proxy[indices]
                batch_X = train_t.backdoor[indices]
                batch_y = train_t.outcome[indices]

                if self.model_name == "naive_neural_net_AY":
                    pred_A = model_a(batch_A.reshape(-1, 1, 64, 64))
                    batch_inputs = torch.cat((pred_A, batch_X), dim=1)
                    pred_y = model(batch_inputs)
                elif self.model_name == "naive_neural_net_AWZY":
                    batch_A = batch_A.reshape(-1, 1, 64, 64)
                    batch_W = batch_A.reshape(-1, 1, 64, 64)
                    pred_A = model_a(batch_A)
                    pred_W = model_w(batch_W)
                    batch_inputs = torch.cat((pred_A, pred_W, batch_Z, batch_X), dim=1)
                    pred_y = model(batch_inputs)
                
                output = loss(pred_y, batch_y)
                
                optimizer.zero_grad()
                optimizer_a.zero_grad()
                if self.model_name == "naive_neural_net_AWZY":
                    optimizer_w.zero_grad()
                output.backward()
                optimizer.step()
                optimizer_a.step()
                if self.model_name == "naive_neural_net_AWZY":
                    optimizer_w.step()

            if self.log_metrics:
                with torch.no_grad():
                    if self.model_name == "naive_neural_net_AY":
                        pred_A = model_a(train_t.treatment.reshape(-1, 1, 64, 64))
                        batch_inputs = torch.cat((pred_A, train_t.backdoor), dim=1)
                        preds_train = model(batch_inputs)
                        pred_A = model_a(val_t.treatment.reshape(-1, 1, 64, 64))
                        batch_inputs = torch.cat((pred_A, val_t.backdoor), dim=1)
                        preds_val = model(batch_inputs)
                    elif self.model_name == "naive_neural_net_AWZY":
                        pred_A = model_a(train_t.treatment.reshape(-1, 1, 64, 64))
                        pred_W = model_w(train_t.outcome_proxy.reshape(-1, 1, 64, 64))
                        batch_inputs = torch.cat((pred_A, pred_W, train_t.treatment_proxy,
                                                    train_t.backdoor), dim=1)
                        preds_train = model(batch_inputs)
                        pred_A = model_a(val_t.treatment.reshape(-1, 1, 64, 64))
                        pred_W = model_w(val_t.outcome_proxy.reshape(-1, 1, 64, 64))
                        batch_inputs = torch.cat((pred_A, pred_W, val_t.treatment_proxy,
                                                    val_t.backdoor), dim=1)
                        preds_val = model(batch_inputs)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = loss(preds_train, train_t.outcome)
                    mse_val = loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)
                    self.train_losses.append(mse_train)
                    self.val_losses.append(mse_val)

        return model

    @staticmethod
    def predict(model_a, model_w, model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch, batch_size=None):

        model_name = model.train_params['name']
        intervention_array_len = test_data_t.treatment.shape[0]
        num_W_test = val_data_t.outcome_proxy.shape[0]

        mean = torch.nn.AvgPool1d(kernel_size=num_W_test, stride=num_W_test)
        with torch.no_grad():
            if batch_size is None:
                # create n_sample copies of each test image (A), and 588 copies of each proxy image (W)
                # reshape test and proxy image to 1 x 64 x 64 (so that the model's conv2d layer is happy)
                test_A = test_data_t.treatment.repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                test_W = val_data_t.outcome_proxy.repeat(intervention_array_len, 1).reshape(-1, 1, 64, 64)
                test_Z = val_data_t.treatment_proxy.repeat(intervention_array_len, 1)
                test_X = val_data_t.backdoor.repeat(intervention_array_len, 1)
                pred_A = model_a(test_A)
                if model_name == "naive_neural_net_AY":
                    batch_inputs = torch.cat((pred_A, test_X), dim=1)
                    E_w_haw = mean(model(batch_inputs).unsqueeze(-1).T)
                elif model_name == "naive_neural_net_AWZY":
                    pred_W = model_w(test_W)
                    batch_inputs = torch.cat((pred_A, pred_W, test_Z, test_X), dim=1)
                    E_w_haw = mean(model(batch_inputs).unsqueeze(-1).T)
            else:
                # the number of A's to evaluate each batch
                a_step = max(1, batch_size//num_W_test)
                E_w_haw = torch.zeros([1, 1, intervention_array_len])
                for a_idx in range(0, intervention_array_len, a_step):
                    temp_A = test_data_t.treatment[a_idx:(a_idx+a_step)].repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                    temp_W = val_data_t.outcome_proxy.repeat(a_step, 1).reshape(-1, 1, 64, 64)
                    temp_Z = val_data_t.treatment_proxy.repeat(a_step, 1)
                    temp_X = val_data_t.backdoor.repeat(a_step, 1)
                    # in this case, we're only predicting for a single A, so we have a ton of W's
                    # therefore, we'll batch this step as well
                    if a_step == 1:
                        model_preds = torch.zeros((temp_A.shape[0]))
                        for temp_idx in range(0, temp_A.shape[0], batch_size):
                            pred_A = model_a(temp_A[temp_idx:temp_idx+batch_size])
                            temp2_X = temp_X[temp_idx:temp_idx+batch_size]
                            if model_name == "naive_neural_net_AY":
                                batch_inputs = torch.cat((pred_A, temp2_X), dim=1)
                                model_preds[temp_idx:(temp_idx+batch_size)] = model(pred_A).squeeze()
                            elif model_name == "naive_neural_net_AWZY":
                                pred_W = model_w(temp_W[temp_idx:temp_idx+batch_size])
                                temp2_Z = temp_Z[temp_idx:temp_idx+batch_size]
                                batch_inputs = torch.cat((pred_A, pred_W, temp2_Z, temp2_X), dim=1)
                                model_preds[temp_idx:(temp_idx+batch_size)] = model(batch_inputs).squeeze()
                        E_w_haw[0, 0, a_idx] = torch.mean(model_preds)
                    else:
                        pred_A = model_a(temp_A)
                        if model_name == "naive_neural_net_AY":
                            batch_inputs = torch.cat((pred_A, temp_X), dim=1)
                            temp_E_w_haw = mean(model(batch_inputs).unsqueeze(-1).T)
                        elif model_name == "naive_neural_net_AWZY":
                            pred_W = model_w(temp_W)
                            batch_inputs = torch.cat((pred_A, pred_W, temp_Z, temp_X), dim=1)
                            temp_E_w_haw = mean(model(batch_inputs).unsqueeze(-1).T)
                        E_w_haw[0, 0, a_idx:(a_idx+a_step)] = temp_E_w_haw[0, 0]

        return E_w_haw.T.squeeze(1).cpu().detach().numpy()

    @staticmethod
    def norm(model_a, model_w, model, val_data_t: PVTrainDataSetTorch, batch_size=None):

        model_name = model.train_params['name']
        n = len(val_data_t.treatment)

        with torch.no_grad():
            if batch_size is None:
                val_A = val_data_t.treatment
                val_W = val_data_t.outcome_proxy
                val_Z = val_data_t.treatment_proxy
                val_X = val_data_t.backdoor
                pred_A = model_a(val_A)
                if model_name == "naive_neural_net_AY":
                    batch_inputs = torch.cat((pred_A, val_X), dim=1)
                elif model_name == "naive_neural_net_AWZY":
                    pred_W = model_w(val_W)
                    batch_inputs = torch.cat((pred_A, pred_W, val_Z, val_X), dim=1)
                model_output = model(batch_inputs)
                norm = torch.sqrt( ( model_output.T @ model_output ) / n )
            else:
                norm2 = 0
                for temp_idx in range(0, n, batch_size):
                    pred_A = model_a(val_A[temp_idx:temp_idx + batch_size])
                    pred_W = model_w(val_W[temp_idx:temp_idx + batch_size])
                    temp_X = val_X[temp_idx:temp_idx + batch_size]
                    temp_Z = val_X[temp_idx:temp_idx + batch_size]
                    if model_name == "naive_neural_net_AY":
                        batch_inputs = torch.cat((pred_A, temp_X), dim=1)
                    elif model_name == "naive_neural_net_AWZY":
                        pred_W = model_w(val_W)
                        batch_inputs = torch.cat((pred_A, pred_W, temp_Z, temp_X), dim=1)
                    model_output = model(batch_inputs)
                    temp_norm2 = ( model_output.T @ model_output )
                    norm2 += temp_norm2

                norm = torch.sqrt( norm2 / n )

            return norm[0,0].cpu().detach().numpy()

class Naive_NN_Trainer_dSpriteExperiment_mono(object):
    def __init__(self, data_config: Dict[str, Any], train_params: Dict[str, Any],
                    analysis_config: Dict[str, Any], random_seed: int,
                    dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):
        self.data_config = data_config
        self.train_params = train_params
        self.analysis_config = analysis_config
        self.device_name = select_device(preferred_device)
        self.data_name = data_config.get("name", None)
        self.n_sample = self.data_config['n_sample']
        self.model_name = self.train_params['name']
        self.n_epochs = self.train_params['n_epochs']
        self.batch_size = self.train_params['batch_size']
        self.weight_decay = self.train_params['weight_decay']
        self.learning_rate = self.train_params['learning_rate']
        self.log_metrics = analysis_config['log_metrics']

        if self.log_metrics and (dump_folder is not None):
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.train_losses = []
            self.val_losses = []

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0):
        if self.model_name == "naive_neural_net_AY":
            # inputs consist of only A
            model = cnn_AY_mono_for_dsprite(train_params=self.train_params)
        elif self.model_name == "naive_neural_net_AWZY":
            # inputs consist of A, W, and Z (and Z is 2-dimensional)
            model = cnn_AWZY_mono_for_dsprite(train_params=self.train_params)
        else:
            raise ValueError(f"name {self.model_name} is not known")

        if self.device_name != "cpu":
            train_t = train_t.to_gpu(self.device_name)
            val_t = val_t.to_gpu(self.device_name)
            model.to(self.device_name)

        # weight_decay implements L2 penalty on weights
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss = nn.MSELoss()

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + self.batch_size]
                batch_y = train_t.outcome[indices]

                if self.model_name == "naive_neural_net_AY":
                    batch_inputs = train_t.treatment[indices]
                    pred_y = model(batch_inputs.reshape(-1, 1, 64, 64))
                if self.model_name == "naive_neural_net_AWZY":
                    batch_A, batch_W, batch_Z, batch_y = ( train_t.treatment[indices], train_t.outcome_proxy[indices],
                                                            train_t.treatment_proxy[indices], train_t.outcome[indices] )
                    batch_A = batch_A.reshape(-1, 1, 64, 64)
                    batch_W = batch_A.reshape(-1, 1, 64, 64)
                    pred_y = model(batch_A, batch_W, batch_Z)
                output = loss(pred_y, batch_y)
                output.backward()
                optimizer.step()

            if self.log_metrics:
                with torch.no_grad():
                    if self.model_name == "naive_neural_net_AY":
                        preds_train = model(train_t.treatment.reshape(-1, 1, 64, 64))
                        preds_val = model(val_t.treatment.reshape(-1, 1, 64, 64))
                    elif self.model_name == "naive_neural_net_AWZY":
                        preds_train = model(train_t.treatment.reshape(-1, 1, 64, 64), train_t.outcome_proxy.reshape(-1, 1, 64, 64), train_t.treatment_proxy)
                        preds_val = model(val_t.treatment.reshape(-1, 1, 64, 64), val_t.outcome_proxy.reshape(-1, 1, 64, 64), val_t.treatment_proxy)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = loss(preds_train, train_t.outcome)
                    mse_val = loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)
                    self.train_losses.append(mse_train)
                    self.val_losses.append(mse_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch, batch_size=None):

        model_name = model.train_params['name']
        intervention_array_len = test_data_t.treatment.shape[0]
        num_W_test = val_data_t.outcome_proxy.shape[0]

        mean = torch.nn.AvgPool1d(kernel_size=num_W_test, stride=num_W_test)
        with torch.no_grad():
            if batch_size is None:
                # create n_sample copies of each test image (A), and 588 copies of each proxy image (W)
                # reshape test and proxy image to 1 x 64 x 64 (so that the model's conv2d layer is happy)
                test_A = test_data_t.treatment.repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                test_W = val_data_t.outcome_proxy.repeat(intervention_array_len, 1).reshape(-1, 1, 64, 64)
                test_Z = val_data_t.treatment_proxy.repeat(intervention_array_len, 1)
                if model_name == "naive_neural_net_AY":
                    E_w_haw = mean(model(test_A).unsqueeze(-1).T)
                elif model_name == "naive_neural_net_AWZY":
                    E_w_haw = mean(model(test_A, test_W, test_Z).unsqueeze(-1).T)
            else:
                # the number of A's to evaluate each batch
                a_step = max(1, batch_size//num_W_test)
                E_w_haw = torch.zeros([1, 1, intervention_array_len])
                for a_idx in range(0, intervention_array_len, a_step):
                    temp_A = test_data_t.treatment[a_idx:(a_idx+a_step)].repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                    temp_W = val_data_t.outcome_proxy.repeat(a_step, 1).reshape(-1, 1, 64, 64)
                    temp_Z = val_data_t.treatment_proxy.repeat(a_step, 1)
                    # in this case, we're only predicting for a single A, so we have a ton of W's
                    # therefore, we'll batch this step as well
                    if a_step == 1:
                        model_preds = torch.zeros((temp_A.shape[0]))
                        for temp_idx in range(0, temp_A.shape[0], batch_size):
                            if model_name == "naive_neural_net_AY":
                                model_preds[temp_idx:(temp_idx+batch_size)] = model(temp_A[temp_idx:temp_idx+batch_size]).squeeze()
                            elif model_name == "naive_neural_net_AWZY":
                                model_preds[temp_idx:(temp_idx+batch_size)] = model(temp_A[temp_idx:temp_idx+batch_size], temp_W[temp_idx:temp_idx+batch_size], temp_Z[temp_idx:temp_idx+batch_size]).squeeze()
                        E_w_haw[0, 0, a_idx] = torch.mean(model_preds)
                    else:
                        if model_name == "naive_neural_net_AY":
                            temp_E_w_haw = mean(model(temp_A).unsqueeze(-1).T)
                        elif model_name == "naive_neural_net_AWZY":
                            temp_E_w_haw = mean(model(temp_A, temp_W, temp_Z).unsqueeze(-1).T)
                        E_w_haw[0, 0, a_idx:(a_idx+a_step)] = temp_E_w_haw[0, 0]

        return E_w_haw.T.squeeze(1).cpu().detach().numpy()

    @staticmethod
    def norm(model_a, model_w, model, val_data_t: PVTrainDataSetTorch, batch_size=None):

        model_name = model.train_params['name']
        n = len(val_data_t.treatment)

        with torch.no_grad():
            if batch_size is None:
                val_A = val_data_t.treatment
                val_W = val_data_t.outcome_proxy
                val_Z = val_data_t.treatment_proxy
                val_X = val_data_t.backdoor
                if model_name == "naive_neural_net_AY":
                    batch_inputs = torch.cat((val_A, val_X), dim=1)
                elif model_name == "naive_neural_net_AWZY":
                    batch_inputs = torch.cat((val_A, val_W, val_Z, val_X), dim=1)
                model_output = model(batch_inputs)
                norm = torch.sqrt( ( model_output.T @ model_output ) / n )
            else:
                norm2 = 0
                for temp_idx in range(0, n, batch_size):
                    temp_A = val_A[temp_idx:temp_idx + batch_size]
                    temp_W = val_W[temp_idx:temp_idx + batch_size]
                    temp_X = val_X[temp_idx:temp_idx + batch_size]
                    temp_Z = val_X[temp_idx:temp_idx + batch_size]
                    if model_name == "naive_neural_net_AY":
                        batch_inputs = torch.cat((temp_A, temp_X), dim=1)
                    elif model_name == "naive_neural_net_AWZY":
                        batch_inputs = torch.cat((temp_A, temp_W, temp_Z, temp_X), dim=1)
                    model_output = model(batch_inputs)
                    temp_norm2 = ( model_output.T @ model_output )
                    norm2 += temp_norm2

                norm = torch.sqrt( norm2 / n )

            return norm[0,0].cpu().detach().numpy()