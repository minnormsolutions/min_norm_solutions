import os.path as op
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn import linear_model

from src.data.ate import (generate_test_data_ate, generate_train_data_ate,
                          generate_val_data_ate, get_preprocessor_ate)
from src.data.ate.data_class import (PVTestDataSetTorch, PVTrainDataSet,
                                     PVTrainDataSetTorch, RHCTestDataSetTorch)
from src.utils.make_AW_test import make_AW_test
from src.utils.select_device import select_device


def twoSLS_RHCexperiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                            analysis_config: Dict[str, Any], one_mdl_dump_dir: Path,
                            random_seed: int = 42, preferred_device: str = "gpu",
                            verbose: int = 0):

    # select optimal device
    device_name = select_device(preferred_device)

    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # load train/val/test data
    train_data = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data = generate_test_data_ate(data_config=data_config,
                                    analysis_config=analysis_config)

    # Combine the train & val splits, then evenly divide in half for first stage & second stage regression
    A = np.concatenate((train_data.treatment, val_data.treatment))
    Z = np.concatenate((train_data.treatment_proxy, val_data.treatment_proxy))
    W = np.concatenate((train_data.outcome_proxy, val_data.outcome_proxy))
    Y = np.concatenate((train_data.outcome, val_data.outcome))
    X = np.concatenate((train_data.backdoor, val_data.backdoor))

    n_sample = len(A)
    permutation = torch.randperm(n_sample)
    first_stage_ix = permutation[: len(permutation) // 2]
    second_stage_ix = permutation[len(permutation) // 2:]

    first_stage_train = PVTrainDataSet(treatment=A[first_stage_ix],
                                       treatment_proxy=Z[first_stage_ix],
                                       outcome_proxy=W[first_stage_ix],
                                       outcome=Y[first_stage_ix],
                                       backdoor=X[first_stage_ix])

    second_stage_train = PVTrainDataSet(treatment=A[second_stage_ix],
                                        treatment_proxy=Z[second_stage_ix],
                                        outcome_proxy=W[second_stage_ix],
                                        outcome=Y[second_stage_ix],
                                        backdoor=X[second_stage_ix])

    # convert datasets to Torch (for GPU runtime)
    first_stage_train_t = PVTrainDataSetTorch.from_numpy(first_stage_train)
    second_stage_train_t = PVTrainDataSetTorch.from_numpy(second_stage_train)
    test_data_t = RHCTestDataSetTorch.from_numpy(test_data)

    # prepare test data on the gpu
    if device_name != "cpu":
        first_stage_train_t = first_stage_train_t.to_gpu(device_name)
        second_stage_train_t = second_stage_train_t.to_gpu(device_name)
        test_data_t = test_data_t.to_gpu(device_name)

    # train 2SLS model (from Miao et al.)
    first_stage_model1 = linear_model.LinearRegression()  # W1 ~ A + X + Z
    first_stage_model2 = linear_model.LinearRegression()  # W2 ~ A + X + Z
    second_stage_model = linear_model.LinearRegression()  # Y ~ A' + X' + \hat{W}

    first_stage_W1 = first_stage_train_t.outcome_proxy[:, 0].reshape(-1, 1)
    first_stage_W2 = first_stage_train_t.outcome_proxy[:, 1].reshape(-1, 1)
    first_stage_features = torch.cat((first_stage_train_t.treatment, first_stage_train_t.backdoor, first_stage_train_t.treatment_proxy), dim=1)
    first_stage_model1.fit(first_stage_features, first_stage_W1)
    first_stage_model2.fit(first_stage_features, first_stage_W2)

    W_hat1 = torch.Tensor(first_stage_model1.predict(
        torch.cat((second_stage_train_t.treatment, second_stage_train_t.backdoor, second_stage_train_t.treatment_proxy), dim=1)))

    W_hat2 = torch.Tensor(first_stage_model2.predict(torch.cat((second_stage_train_t.treatment, second_stage_train_t.backdoor,
                                                                second_stage_train_t.treatment_proxy), dim=1)))

    W_hat = torch.cat((W_hat1, W_hat2), dim=1)
    second_stage_model.fit(torch.cat((second_stage_train_t.treatment, second_stage_train_t.backdoor, W_hat), dim=1),
                           second_stage_train_t.outcome)

    # Create a 3-dim array with shape [n_treatments, n_samples, len(A) + len(W) + len(X)]
    # The first axis contains the two values of do(A): 0 and 1
    # The last axis contains W, X, needed for the model's forward pass
    n_treatments = len(test_data_t.treatment)
    n_samples = len(test_data_t.outcome_proxy)
    tempA = test_data_t.treatment.unsqueeze(-1).expand(-1, n_samples, -1)
    tempW = test_data_t.outcome_proxy.unsqueeze(0).expand(n_treatments, -1, -1)
    tempX = test_data_t.backdoor.unsqueeze(0).expand(n_treatments, -1, -1)
    model_inputs_test = torch.dstack((tempA, tempW, tempX))

    # get model predictions for A=0 and A=1 on test data
    EY_noRHC = np.mean(second_stage_model.predict(model_inputs_test[0, :, :]))
    EY_RHC = np.mean(second_stage_model.predict(model_inputs_test[1, :, :]))
    pred = [EY_noRHC, EY_RHC]
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if hasattr(test_data, 'structural'):
        # test_data.structural is equivalent to EY_doA
        np.testing.assert_array_equal(pred.shape, test_data.structural.shape)
        oos_loss = np.mean((pred - test_data.structural) ** 2)
    else:
        oos_loss = None


def twoSLS_Demandexperiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                                analysis_config: Dict[str, Any], one_mdl_dump_dir: Path,
                                random_seed: int = 42, preferred_device: str = "gpu",
                                verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    first_stage_train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    second_stage_train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed + 2)
    val_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data_org = generate_test_data_ate(data_config=data_config,
                                        analysis_config=analysis_config)

    # preprocess data
    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    first_stage_train_data = preprocessor.preprocess_for_train(first_stage_train_data_org)
    second_stage_train_data = preprocessor.preprocess_for_train(second_stage_train_data_org)
    first_stage_train_t = PVTrainDataSetTorch.from_numpy(first_stage_train_data)
    second_stage_train_t = PVTrainDataSetTorch.from_numpy(second_stage_train_data)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    val_data = preprocessor.preprocess_for_train(val_data_org)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    # train 2SLS model (from Miao et al.)
    first_stage_model = linear_model.LinearRegression()  # W ~ A + Z
    second_stage_model = linear_model.LinearRegression()  # Y ~ A + \hat{W}

    first_stage_W = first_stage_train_t.outcome_proxy.reshape(-1, 1)
    first_stage_features = torch.cat((first_stage_train_t.treatment, first_stage_train_t.treatment_proxy), dim=1)
    first_stage_model.fit(first_stage_features, first_stage_W)
    W_hat = torch.Tensor(first_stage_model.predict(
        torch.cat((second_stage_train_t.treatment, second_stage_train_t.treatment_proxy), dim=1)))
    second_stage_model.fit(torch.cat((second_stage_train_t.treatment, W_hat), dim=1),
                           second_stage_train_t.outcome.reshape(-1, 1))

    AW_test = make_AW_test(test_data_t, val_data_t)

    # get model predictions on do(A) intervention values
    pred = [np.mean(second_stage_model.predict(AW_test[i, :, :])) for i in range(AW_test.shape[0])]
    
    # np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    # if test_data.structural is not None:
    #     # test_data_org.structural is equivalent to EY_doA
    #     oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
    #     if data_config["name"] in ["kpv", "deaner"]:
    #         oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))
    return pred, None


def twoSLS_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                        analysis_config: Dict[str, Any], one_mdl_dump_dir: Path,
                        random_seed: int = 42, preferred_device: str = "gpu",
                        verbose: int = 0):
    data_name = data_config.get("name", None)

    if data_name.lower() == 'demand':
        return twoSLS_Demandexperiment(data_config, model_param, analysis_config,
                                        one_mdl_dump_dir, random_seed)
    elif data_name.lower() == 'rhc':
        return twoSLS_RHCexperiment(data_config, model_param, one_mdl_dump_dir, random_seed)
    else:
        raise KeyError(f"The `name` key in config.json was {data_name} but must be one of [demand, rhc]")
