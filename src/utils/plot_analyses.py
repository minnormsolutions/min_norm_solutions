import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(file_dir, target_dir = None, linear_bounds = None):
    if target_dir is None:
        target_dir = file_dir
    
    if linear_bounds is None:
        linear_bounds = (None, None)
    elif isinstance(linear_bounds, str) :
        linear_bounds = json.loads(linear_bounds)

    file_location = file_dir + "results.csv"
    target_location = target_dir + "individual_prediction_curves.png"
    target_location_log = target_dir + "individual_prediction_curves_log.png"
    predictions = pd.read_csv( file_location, index_col = 0 )
    prediction_curves = predictions.iloc[:-5, :-2]
    A = prediction_curves.columns
    index = prediction_curves.index

    fig, ax = plt.subplots( layout = "constrained" )

    for i, row in enumerate(prediction_curves.index):
            ax.plot( A, prediction_curves.loc[row], label = row )

    ax.set_title("Prediction Curves")
    ax.set_xlabel("A")
    ax.set_ylabel("Y")
    ax.set_ybound(lower = linear_bounds[0], upper = linear_bounds[1])
    ax.legend(loc = "center right", bbox_to_anchor = (1.3, 0.5))
    # plt.show()

    fig.savefig(target_location, bbox_inches="tight")

    ax.set_ybound(lower=None, upper=None)
    ax.set_yscale("log")
    # plt.show()

    fig.savefig(target_location_log, bbox_inches="tight")

def plot_prediction_curves(file_dir, target_dir = None, lambda2 = False,
                                linear_bounds = None):
    if target_dir is None:
        target_dir = file_dir

    if linear_bounds is None:
        linear_bounds = (None, None)
    elif isinstance(linear_bounds, str) :
        linear_bounds = json.loads(linear_bounds)

    file_location = file_dir + "predictions.csv"
    target_location = target_dir + "prediction_curves.png"
    target_location_log = target_dir + "prediction_curves_log.png"
    predictions = pd.read_csv(file_location, index_col = 0)
    A = predictions.columns
    index = predictions.index
    
    fig, ax = plt.subplots( layout = "constrained" )

    if lambda2:
        ax.plot( A, predictions.iloc[0], label = index[0] )
        ax.plot( A, predictions.iloc[1], label = r"$\lambda$" + "=0" )
        for i, row in enumerate(predictions.index[2:]):
            ax.plot( A, predictions.loc[row],
                    label = r"$\lambda$" + f"={10**(i-10):.0e}" )
    else:
        for i, row in enumerate(predictions.index):
            ax.plot( A, predictions.loc[row], label = row )

    ax.set_title("Prediction Curves")
    ax.set_xlabel("A")
    ax.set_ylabel("Y")
    ax.set_ybound(lower = linear_bounds[0], upper = linear_bounds[1])
    ax.legend(loc = "center right", bbox_to_anchor = (1.3, 0.5))
    # plt.show()

    fig.savefig(target_location, bbox_inches="tight")

    ax.set_ybound(lower=None, upper=None)
    ax.set_yscale("log")
    # plt.show()

    fig.savefig(target_location_log, bbox_inches="tight")

    return None

def plot_summary_data(file_dir, target_dir = None, lambda2 = False):
    if target_dir is None:
        target_dir = file_dir
    file_location = file_dir + "summary_stats.csv"
    target_location = target_dir + "summary_plots.png"
    summary_stats = pd.read_csv(file_location, index_col=0)
    summary_stats.drop( columns = "Bias", inplace=True)
    stat_names = summary_stats.columns
    index = summary_stats.index
    x0 = [ x for x in range(len(index)) ]

    fig, axs = plt.subplots( nrows = 2, ncols = 2, layout = "constrained" )

    for i, ax in enumerate(axs.flat):
        ax.plot( x0, summary_stats[stat_names[i]], label = stat_names[i] )
        ax.set_title(stat_names[i])
        if lambda2:
            ax.set_xlabel(r"$\lambda$")
            ax.set_xticks([0] + [ 2 + 3*x for x in range(4)])
            ax.set_xticklabels(["0"] + [f"{10**x:.0e}" for x in range(-9,1,3)])
        else:
            col_loc = index[0].find(":")
            vals = []
            for i in range(len(index)):
                vals.append( index[i][(col_loc + 1):] )
            name = index[0][:col_loc]
            div = ( len(index) - 1 ) // 3
            ticks = [ div * x for x in range(4) ]
            ax.set_xlabel(name)
            ax.set_xticks(ticks)
            ax.set_xticklabels([ vals[x] for x in ticks])
        ax.set_yscale("log")
    # plt.show()

    fig.savefig(target_location, bbox_inches="tight")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir")
    parser.add_argument("--target_dir")
    parser.add_argument("--lambda2")
    parser.add_argument("--linear_bounds")
    args = parser.parse_args()

    results_file = os.path.join(args.file_dir, "results.csv")
    prediction_file = os.path.join(args.file_dir, "prediction.csv")
    summary_file = os.path.join(args.file_dir, "summary_stats.csv")

    if os.path.isfile(results_file):
        plot_results( args.file_dir, args.target_dir, args.linear_bounds )
    if os.path.isfile(prediction_file):
        plot_prediction_curves( args.file_dir, args.target_dir, args.lambda2,
                                    args.linear_bounds )
    if os.path.isfile(summary_file):
        plot_summary_data( args.file_dir, args.target_dir, args.lambda2 )