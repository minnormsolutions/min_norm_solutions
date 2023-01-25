import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.ate import generate_test_data_ate
from src.models.CEVAE.trainer import cevae_experiments
from src.models.DFPV.trainer import dfpv_experiments
from src.models.kernelPV.model import kpv_experiments
from src.models.linear_regression.linear_reg_experiments import \
    linear_reg_demand_experiments
from src.models.naive_neural_net.naive_nn_experiments import \
    naive_nn_experiments
from src.models.NMMR.NMMR_experiments import NMMR_experiments
from src.models.PMMR.model import pmmr_experiments
from src.models.twoSLS.twoSLS_experiments import twoSLS_experiment
from src.utils import grid_search_dict
from src.utils.aggregate_data import aggregate_data
from src.utils.hyperparameter_utils import compare_hyperparameters
from src.utils.dataframe_utils import mse, row_mse
from src.utils.plot_analyses import plot_summary_data

logger = logging.getLogger()


def get_experiment(mdl_name: str):
    if mdl_name == "kpv":
        return kpv_experiments
    elif mdl_name == "dfpv":
        return dfpv_experiments
    elif mdl_name == "pmmr":
        return pmmr_experiments
    elif mdl_name == "cevae":
        return cevae_experiments
    elif mdl_name == "nmmr":
        return NMMR_experiments
    elif mdl_name in ["linear_regression_AY", "linear_regression_AWZY",
                        "linear_regression_AY2", "linear_regression_AWZY2"]:
        return linear_reg_demand_experiments
    elif ( mdl_name == "naive_neural_net_AY"
            or mdl_name == "naive_neural_net_AWZY" ):
        return naive_nn_experiments
    elif mdl_name == "twoSLS":
        return twoSLS_experiment
    else:
        raise ValueError(f"name {mdl_name} is not known")


def experiments(configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int = 1,
                preferred_device: str = "gpu"):
    
    data_config = configs["data"]
    model_config = configs["model"]
    if "analysis" in configs:
        analysis_config = configs["analysis"]
    else:
        analysis_config = {}
    n_repeat: int = configs["n_repeat"]

    # Fill in missing keys

    if "compute_statistics" not in analysis_config:
        analysis_config["compute_statistics"] = False
    if "log_metrics" not in analysis_config:
        analysis_config["log_metrics"] = False
    if "intervention_array" in analysis_config:
        analysis_config["n_interventions"] = len(
                                    analysis_config["intervention_array"])
    else:
        if "n_interventions" not in analysis_config:
            analysis_config["n_interventions"] = 11
        analysis_config["intervention_array"] = np.linspace(10, 30,
                                    analysis_config["n_interventions"] )

    if num_cpus <= 1 and n_repeat <= 1:
        verbose: int = 2
    else:
        verbose: int = 0

    i = 0
    experiment = get_experiment(model_config["name"])
    for dump_name, env_param in grid_search_dict(data_config):
        if dump_name != "one":
            one_dump_dir = os.path.join(dump_dir, dump_name)
            os.mkdir(one_dump_dir)
        else:
            one_dump_dir = dump_dir
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = os.path.join(one_dump_dir, mdl_dump_name)
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir

            predictions = np.zeros( ( n_repeat + 6,
                analysis_config["n_interventions"] + 2 ) )
            predictions = pd.DataFrame(predictions)
            predictions.columns = ( [ x for x in
                        analysis_config["intervention_array"] ]
                            + ["Norm/Mean"] + ["MSE"] )
            predictions.index = ( [ x for x in range(n_repeat) ]
                            + ["do(A)"] + ["Mean"] + ["Bias"] + ["Bias^2"]
                            + ["Variance"] + ["MSE"] )

            if analysis_config["log_metrics"]:
                train_metrics_ls = []
                for idx in range(n_repeat):
                    preds, norm, train_metrics = experiment(env_param, mdl_param,
                            analysis_config, one_mdl_dump_dir, idx, preferred_device, verbose)
                    predictions.iloc[idx, :-2] = preds
                    predictions.iloc[idx, -2] = norm
                    train_metrics['rep_ID'] = idx
                    train_metrics_ls.append(train_metrics)

                metrics_df = pd.concat(train_metrics_ls).reset_index()
                metrics_df.rename(columns={'index': 'epoch_num'}, inplace=True)
                metrics_df.to_csv(os.path.join(one_mdl_dump_dir, "train_metrics.csv"), index=False)
            
            else:
                for idx in range(n_repeat):
                    preds, norm = experiment(env_param, mdl_param, analysis_config,
                            one_mdl_dump_dir, idx, preferred_device, verbose)
                    predictions.iloc[idx, :-2] = preds
                    predictions.iloc[idx, -2] = norm

            do_A = generate_test_data_ate(data_config=env_param,
                            analysis_config=analysis_config).structural
            do_A = do_A.flatten()
            predictions.iloc[n_repeat, :-2] = do_A
            predictions.iloc[n_repeat + 1] = predictions.iloc[:n_repeat].mean()
            predictions.iloc[n_repeat + 2, :-2] = (
                                    predictions.iloc[n_repeat + 1, :-2]
                                        - predictions.iloc[n_repeat, :-2]
            )
            predictions.iloc[n_repeat + 3, :-2] = predictions.iloc[n_repeat + 2, :-2]**2
            predictions.iloc[n_repeat + 4, :-2] = predictions.iloc[:n_repeat, :-2].var()
            predictions.iloc[n_repeat + 5, :-2] = (
                predictions.iloc[:(n_repeat+1), :-2].apply(mse, args=(n_repeat,))
            )
            predictions.iloc[n_repeat + 2:, -2] = (
                    predictions.iloc[n_repeat+2:, :-2].mean(axis = 1)
            )
            predictions.iloc[:n_repeat, -1] = row_mse( predictions, n_repeat, -2 )
            predictions.iloc[n_repeat + 1, -1] = predictions.iloc[:n_repeat, -1].mean()
            predictions.iloc[-1, -1] = predictions.iloc[n_repeat + 1, -1]
            predictions.iloc[-2, -1] = predictions.iloc[n_repeat + 1, -2]

            predictions.to_csv(os.path.join(one_mdl_dump_dir, "results.csv"))

            i += n_repeat
            print(i)

    if analysis_config["compute_statistics"]:
        if dump_name == "one":
            aggregate_data(dump_dir, analyses = "all")
        else:
            aggregate_data(dump_dir, analyses = "summary_stats")
        if analysis_config["log_metrics"]:
            compare_hyperparameters(dump_dir)
    if analysis_config.get("make_plots", False):
        plot_summary_data(file_dir = os.path.join(dump_dir, "analyses/"))