import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data.ate import generate_test_data_ate
from src.utils import grid_search_dict


def aggregate_data(dump_dir:str, out_dir:str = None,
                    analyses = ["predictions", "summary_stats"]):
    
    # id_str = dump_dir[ (dump_dir.find("_") + 1):]
    config_path = os.path.join(dump_dir, 'configs.json')
    with open(config_path) as config_file:
        config = json.load(config_file)
    n_repeat = config["n_repeat"]
    data_config = config["data"]
    model_config = config["model"]
    if "analysis" in config:
        analysis_config = config["analysis"]
    else:
        analysis_config = {}
    
    if analyses == "all":
        analyses = ["predictions", "summary_stats"]
    if isinstance(analyses, str):
        analyses = [analyses]
    analysis_config["analysis_list"] = analyses

    for item in analysis_config["analysis_list"]:
        analysis_config[item] = True

    for key in ["predictions", "summary_stats"]:
        if key not in analysis_config:
            analysis_config[key] = False

    if analysis_config["predictions"] or analysis_config["summary_stats"]:
        if out_dir is None:
            out_dir = os.path.join(dump_dir, "analyses")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
    else:
        return None

    if "intervention_array" in analysis_config:
        analysis_config["n_interventions"] = len(
                                    analysis_config["intervention_array"])
    else:
        if "n_interventions" not in analysis_config:
            analysis_config["n_interventions"] = 11
        analysis_config["intervention_array"] = np.linspace(10, 30,
                                    analysis_config["n_interventions"] )

    n_interventions = analysis_config["n_interventions"]

    if analysis_config["predictions"]:
        do_A = generate_test_data_ate(data_config=data_config,
                    analysis_config = analysis_config).structural
        predictions = [ do_A.flatten() ]

    if analysis_config["summary_stats"]:
        summary_stats = []
    
    index_names = []

    i = 0
    for dump_name, env_param in grid_search_dict(data_config):
        if dump_name != "one":
            one_dump_dir = os.path.join(dump_dir, dump_name)
        else:
            one_dump_dir = dump_dir
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = os.path.join(one_dump_dir, mdl_dump_name)
            else:
                one_mdl_dump_dir = one_dump_dir
            if os.path.exists(one_mdl_dump_dir):
                input_df = pd.read_csv(os.path.join(one_mdl_dump_dir, 'results.csv'),
                                        index_col = 0)

                if analysis_config["predictions"]:
                    predictions.append(input_df.loc["Mean"][:-2])
                if analysis_config["summary_stats"]:
                    summary_stats.append(input_df.iloc[ -5:, -2])
                if dump_name == "one":
                    index_names.append(mdl_dump_name)
                else:
                    if mdl_dump_name == "one":
                        index_names.append(dump_name)
                    else:
                        index_names.append(dump_name + mdl_dump_name)
                
                i += 1

    if analysis_config["predictions"]:
        predictions = np.vstack(predictions)
        predictions = pd.DataFrame(predictions)
        predictions.columns = [ x for
                        x in analysis_config["intervention_array"] ]
        predictions.index = ( ["do(A)"] + index_names )
        predictions.to_csv(os.path.join(out_dir, "predictions.csv"))

    if analysis_config["summary_stats"]:
        summary_stats = pd.DataFrame(summary_stats)
        summary_stats.columns = ( ["Norm"] + ["Bias"] + ["Bias^2"]
                                    + ["Variance"] + ["MSE"] )
        summary_stats.index = index_names
        summary_stats.to_csv(os.path.join(out_dir, "summary_stats.csv"))

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir')
    parser.add_argument('--out_dir')
    parser.add_argument('--analyses')
    args = parser.parse_args()

    aggregate_data(args.dump_dir, args.out_dir, args.analyses)

    # results_df, id_str, n_repeat = get_hyperparameter_results_dataframe(args.dump_dir)
    # results_df.to_csv(op.join(args.out_dir, f"hp_results_x{n_repeat}_{id_str}.csv"), index=False,
    #                     float_format="%.2e")
    # results_df.to_pickle(op.join(args.out_dir, "hp_results.pkl"))
