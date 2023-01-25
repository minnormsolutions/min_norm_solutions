import os.path as op
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from src.data.ate import (generate_test_data_ate, generate_train_data_ate,
                          generate_val_data_ate, get_preprocessor_ate)
from src.data.ate.data_class import (PVTestDataSetTorch, PVTrainDataSetTorch,
                                     RHCTestDataSetTorch)
from src.models.NMMR.NMMR_trainers import (NMMR_Trainer_DemandExperiment,
                                           NMMR_Trainer_dSpriteExperiment,
                                           NMMR_Trainer_dSpriteExperiment_mono,
                                           NMMR_Trainer_RHCExperiment)


def NMMR_experiments(data_config: Dict[str, Any], model_config: Dict[str, Any],
                    analysis_config: Dict[str, Any], one_mdl_dump_dir: Path,
                    random_seed: int = 42, preferred_device: str = "gpu",
                    verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    train_data = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data = generate_test_data_ate(data_config=data_config,
                                    analysis_config=analysis_config)

    # convert datasets to Torch (for GPU runtime)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    data_name = data_config.get("name", None)
    if data_name in ['dsprite', 'demand']:
        test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    elif data_name == 'rhc':
        test_data_t = RHCTestDataSetTorch.from_numpy(test_data)
    else:
        raise KeyError(f"Your data config contained name = {data_name}, but must be one of [dsprite, demand, rhc]")

    # retrieve the trainer for this experiment
    if data_name == "demand":
        trainer = NMMR_Trainer_DemandExperiment(data_config, model_config,
                    analysis_config, random_seed, one_mdl_dump_dir, preferred_device)
    elif data_name == 'rhc':
        trainer = NMMR_Trainer_RHCExperiment(data_config, model_config,
                    analysis_config, random_seed, one_mdl_dump_dir, preferred_device)
    elif data_name == "dsprite":
        if model_config['cnn_type'] == "monolithic":
            trainer = NMMR_Trainer_dSpriteExperiment_mono(data_config, model_config,
                        analysis_config, random_seed, one_mdl_dump_dir, preferred_device)
        else:
            trainer = NMMR_Trainer_dSpriteExperiment(data_config, model_config,
                        analysis_config, random_seed, one_mdl_dump_dir, preferred_device)

    # train model
    if ( ( data_name == "dsprite" ) and ( model_config['cnn_type'] != "monolithic" ) ):
        model_a, model_w, model = trainer.train(train_t, val_data_t, verbose)
    else:
        model = trainer.train(train_t, val_data_t, verbose)

    # prepare test data on the gpu
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
    # if hasattr(test_data, 'structural'):
    #     # test_data.structural is equivalent to EY_doA
    #     np.testing.assert_array_equal(pred.shape[0], test_data.structural.shape[0])
    #     np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.structural.txt"), pred)
    #     oos_loss = np.mean((pred - test_data.structural) ** 2)
    # else:
    #     oos_loss = None

    if trainer.log_metrics:
        return pred, norm, pd.DataFrame(
            data={
                    'causal_penalized_loss_train': torch.Tensor(
                            trainer.causal_penalized_losses_train[-50:],
                                device="cpu").numpy(),
                    'causal_unpenalized_loss_train': torch.Tensor(
                            trainer.causal_unpenalized_losses_train[-50:],
                                device="cpu").numpy(),
                    'causal_norm2_train': torch.Tensor(
                            trainer.causal_norm2s_train[-50:],
                                device="cpu").numpy(),
                    'causal_penalized_loss_val': torch.Tensor(
                            trainer.causal_penalized_losses_val[-50:],
                                device="cpu").numpy(),
                    'causal_unpenalized_loss_val': torch.Tensor(
                            trainer.causal_unpenalized_losses_val[-50:],
                                device="cpu").numpy(),
                    'causal_norm2_val': torch.Tensor(
                            trainer.causal_norm2s_val[-50:],
                                device="cpu").numpy(),
                })
    else:
        return pred, norm
