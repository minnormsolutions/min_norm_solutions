import os.path as op
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from src.data.ate import (generate_test_data_ate, generate_train_data_ate,
                          generate_val_data_ate, get_preprocessor_ate)
from src.data.ate.data_class import PVTestDataSetTorch, PVTrainDataSetTorch
from src.models.naive_neural_net.naive_nn_trainers import (
    Naive_NN_Trainer_DemandExperiment, Naive_NN_Trainer_dSpriteExperiment,
    Naive_NN_Trainer_dSpriteExperiment_mono)
from src.utils.make_AWZ_test import make_AWZ_test


def naive_nn_experiments(data_config: Dict[str, Any], model_config: Dict[str, Any],
                           analysis_config: Dict[str, Any], one_mdl_dump_dir: Path,
                           random_seed: int = 42, preferred_device: str = "gpu",
                           verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data_org = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data_org = generate_test_data_ate(data_config=data_config,
                                        analysis_config=analysis_config)

    # preprocess data
    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    val_data = preprocessor.preprocess_for_train(val_data_org)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    data_name = data_config.get("name", None)

    # retrieve the trainer for this experiment
    if data_name == "demand":
        trainer = Naive_NN_Trainer_DemandExperiment(data_config, model_config,
                    analysis_config, random_seed, one_mdl_dump_dir, preferred_device)
    elif data_name == "dsprite":
        if model_config['cnn_type'] == "monolithic":
            trainer = Naive_NN_Trainer_dSpriteExperiment_mono(data_config, model_config,
                    analysis_config, random_seed, one_mdl_dump_dir, preferred_device)
        else:
            trainer = Naive_NN_Trainer_dSpriteExperiment(data_config, model_config,
                    analysis_config, random_seed, one_mdl_dump_dir, preferred_device)

    # train model
    if ( ( data_name == "dsprite" ) and ( model_config['cnn_type'] != "monolithic" ) ):
        model_a, model_w, model = trainer.train(train_t, val_data_t, verbose)
    else:
        model = trainer.train(train_t, val_data_t, verbose)

    # prepare test and val data on the gpu
    if trainer.device_name != "cpu":
        test_data_t = test_data_t.to_gpu(trainer.device_name)
        val_data_t = val_data_t.to_gpu(trainer.device_name)

    if data_name == "demand":
        pred = trainer.predict(model, test_data_t, val_data_t)
        norm = trainer.norm(model, val_data_t)
    elif data_name == "rhc":
        pred = trainer.predict(model, test_data_t)
        norm = trainer.norm(model, val_data_t)
    elif data_name == "dsprite":
        if model_config['cnn_type'] != "monolithic":
            pred = trainer.predict(model_a, model_w, model,
                test_data_t, val_data_t,
                batch_size=model_config.get('val_batch_size', None))
            norm = trainer.norm(model_a, model_w, model, val_data_t,
                batch_size=model_config.get('val_batch_size', None))
        else:
            pred = trainer.predict(model, test_data_t, val_data_t,
                batch_size=model_config.get('val_batch_size', None))
            norm = trainer.norm(model, val_data_t,
                batch_size=model_config.get('val_batch_size', None))
    
    # np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    # if test_data.structural is not None:
    #     # test_data_org.structural is equivalent to EY_doA
    #     oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
    #     if data_config["name"] in ["kpv", "deaner"]:
    #         oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))

    if trainer.log_metrics:
        return pred, norm, pd.DataFrame(
            data={'obs_MSE_train': torch.Tensor(trainer.train_losses[-50:], device="cpu").numpy(),
                  'obs_MSE_val': torch.Tensor(trainer.val_losses[-50:], device="cpu").numpy()})
    else:
        return pred, norm