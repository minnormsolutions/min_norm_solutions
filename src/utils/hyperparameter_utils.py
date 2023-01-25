import argparse
import json
import os

import pandas as pd
# from tensorflow.python.summary.summary_iterator import summary_iterator

from src.utils import grid_search_dict


def compare_hyperparameters(dump_dir, out_dir = None,
                                loss_name = "causal_loss_val"):
    if out_dir is None:
        out_dir = os.path.join(dump_dir, "analyses")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    # id_str = dump_dir[ (dump_dir.find("_") + 1):]
    config_path = os.path.join(dump_dir, 'configs.json')
    with open(config_path) as config_file:
        config = json.load(config_file)
    n_repeat = config["n_repeat"]
    data_config = config["data"]
    model_config = config["model"]

    results = pd.DataFrame()

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
            
            combined_param_dict = env_param | mdl_param
            if os.path.exists(one_mdl_dump_dir):
                # tensorboard_dir = os.path.join(one_mdl_dump_dir, 'tensorboard_log')
                input_df = pd.read_csv(os.path.join(one_mdl_dump_dir, 'train_metrics.csv'))
                temp = input_df.groupby('rep_ID').mean()

                if loss_name not in input_df.columns:
                    if "causal_loss_val" in input_df.columns:
                        loss_name = "causal_loss_val"
                    elif "causal_unpenalized_loss_val" in input_df.columns:
                        loss_name = "causal_unpenalized_loss_val"
                temp = input_df.groupby('rep_ID').mean()
                mean_avg_causal_val_loss = temp.loc[:, loss_name].mean()
                median_avg_causal_val_loss = temp.loc[:, loss_name].median()
                max_avg_causal_val_loss = temp.loc[:, loss_name].max()
                combined_param_dict['mean_avg_val_loss'] = mean_avg_causal_val_loss
                combined_param_dict['median_avg_val_loss'] = median_avg_causal_val_loss
                combined_param_dict['max_avg_val_loss'] = max_avg_causal_val_loss
                temp = pd.DataFrame([combined_param_dict])
                results = pd.concat([results, temp])

    results.to_csv(os.path.join(out_dir, "hp_results.csv"))

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir')
    parser.add_argument('--out_dir')
    parser.add_argument('--loss_name')
    args = parser.parse_args()

    compare_hyperparameters(args.dump_dir, args.out_dir)

    # if args.out_dir is None:
    #         args.out_dir = os.path.join(args.dump_dir, "analyses")
    #         if not os.path.exists(args.out_dir):
    #             os.mkdir(args.out_dir)
    #         results, id_str, n_repeat = compare_hyperparameters(args.dump_dir)
    #         results.to_csv(os.path.join(args.out_dir, "hp_results.csv"), index=False,
    #                     float_format="%.2e")
    # else:
    #     results, id_str, n_repeat = compare_hyperparameters(args.dump_dir)
    #     results.to_csv(os.path.join(args.out_dir, f"hp_results_x{n_repeat}_{id_str}.csv"), index=False,
    #                         float_format="%.2e")
    # results.to_pickle(op.join(args.out_dir, "hp_results.pkl"))
