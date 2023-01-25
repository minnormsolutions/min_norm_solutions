from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

from src.data.ate import (generate_test_data_ate, generate_train_data_ate,
                          get_preprocessor_ate)
from src.data.ate.data_class import (PVTestDataSet, PVTestDataSetTorch,
                                     PVTrainDataSet, PVTrainDataSetTorch,
                                     split_train_data)
from src.models.CEVAE.model import CEVAEModel
from src.models.CEVAE.nn_structure import build_extractor
from src.utils.select_device import select_device

logger = logging.getLogger()


class CEVAETrainer(object):

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 dump_folder: Optional[Path] = None, preferred_device: str = "gpu"):
        self.data_config = data_configs
        self.device_name = select_device(preferred_device)
        if self.device_name != "cpu":
            logger.info("gpu mode")
        # configure training params
        self.n_epoch: int = train_params["n_epoch"]
        self.weight_decay = train_params["weight_decay"]
        self.n_learning_sample = train_params["n_learning_sample"]
        self.early_stop = train_params["early_stop"]

        # build networks
        self.distribution = build_extractor(data_configs["name"],
                                            hidden_dim=train_params["hidden_dim"],
                                            n_sample=self.n_learning_sample)
        if self.device_name != "cpu":
            self.distribution = self.distribution.to(self.device_name)

        self.opt = torch.optim.Adamax(self.distribution.parameters(),
                                      weight_decay=self.weight_decay,
                                      lr=0.01)

        self.scheduler = ExponentialLR(self.opt, gamma=0.99)

    def train(self, train_data: PVTrainDataSet, test_data_t, verbose: int = 0):
        train_data, val_data = split_train_data(train_data, 0.9)
        train_data_t = PVTrainDataSetTorch.from_numpy(train_data)
        val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

        if self.device_name != "cpu":
            train_data_t = train_data_t.to_gpu(self.device_name)
            val_data_t = val_data_t.to_gpu(self.device_name)

        proxy = torch.cat([train_data_t.outcome_proxy, train_data_t.treatment_proxy], dim=1)
        val_proxy = torch.cat([val_data_t.outcome_proxy, val_data_t.treatment_proxy], dim=1)
        early_stop_count = 0
        min_loss = np.inf
        for t in range(self.n_epoch):
            self.opt.zero_grad()
            loss = self.distribution(proxy=proxy,
                                     treatment=train_data_t.treatment,
                                     outcome=train_data_t.outcome)
            loss.backward()
            self.opt.step()
            self.scheduler.step()
            with torch.no_grad():
                val_loss = self.distribution(proxy=val_proxy,
                                             treatment=val_data_t.treatment,
                                             outcome=val_data_t.outcome).data.item()
                if val_loss < min_loss:
                    min_loss = val_loss
                    early_stop_count = 0
                else:
                    early_stop_count += 1

            if early_stop_count >= self.early_stop:
                break

            if verbose >= 2:
                logger.info(f"{t}-Iteration: VAE: {loss.item()}")

            if verbose >= 2 and t % 10 == 0:
                with torch.no_grad():
                    mdl = CEVAEModel(self.distribution)
                    mdl.fit(proxy, train_data_t.treatment, train_data_t.outcome,
                            10)
                    oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
                    logger.info(f"{t}-Iteration: oos_loss: {oos_loss}")

            if verbose >= 2 and t % 10 == 0:
                with torch.no_grad():
                    logger.info(self.distribution.q_z_xty(val_proxy,
                                                          val_data_t.treatment,
                                                          val_data_t.outcome).mean)

        mdl = CEVAEModel(self.distribution)
        mdl.fit(proxy, train_data_t.treatment, train_data_t.outcome, 10)
        return mdl


def cevae_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                        analysis_config: Dict[str, Any], one_mdl_dump_dir: Path,
                        random_seed: int = 42, preferred_device: str = "gpu",
                        verbose: int = 0):
    dump_dir = one_mdl_dump_dir.joinpath(f"{random_seed}")
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config,
                                        analysis_config=analysis_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    torch.manual_seed(random_seed)
    trainer = CEVAETrainer(data_config, model_param, dump_dir, preferred_device)

    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    if trainer.device_name != "cpu":
        test_data_t = test_data_t.to_gpu(trainer.device_name)

    mdl = trainer.train(train_data, test_data_t, verbose)
    pred: np.ndarray = mdl.predict_t(test_data_t.treatment).data.flatten().cpu().detach().numpy()
    pred = preprocessor.postprocess_for_prediction(pred)
    # oos_loss = 0.0
    # if test_data.structural is not None:
    #     oos_loss: float = np.mean((pred - test_data_org.structural) ** 2)
    #     if data_config["name"] in ["kpv", "deaner"]:
    #         oos_loss = np.mean(np.abs(pred - test_data_org.structural))
    # np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    return pred, None
