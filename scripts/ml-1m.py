import argparse
from argparse import Namespace
from dataclasses import dataclass
import os
import random

import torch
import numpy as np
from sklearn.metrics import brier_score_loss

from src.bounds import *
from src.utils import predict, calibrator
from src.metric import balanced_accuracy
from src.metric import fdr
from src.loss_fns import *
from src.exp_tools import *

from matplotlib import style
plt.style.use('seaborn-v0_8')

import matplotlib.pyplot as plt


def main(args):
    
    set_all_seeds(args.seed)
    args.n_val = args.max_full_pop

    thresholds = get_thresholds(args)
    loss = torch.load("../data/ml-1m/loss_matrix.pt")
    g = torch.load("../data/ml-1m/group_info.pt")
    
    print("loss", loss.shape)
    print("g", g.shape)
    
    args.var_key, args.interest_vars, args.no_groups = load_group_info(args)    

    load_configs(args)
    
    args.correction = args.delta/(args.num_hypotheses*args.no_bounds)
    
    args.split_interval = (args.beta_max_2 is not None)
    
    loss_fns = load_losses(args)

    print("t", thresholds.shape, "loss shape", loss.shape)
    print("group stuff", args.var_key, args.interest_vars, args.no_groups)
    print("loss fns", loss_fns)
    print("args", args)
    print()

    rows = []
    
    for trial_idx in range(args.no_trials):
        
        print()
        print("Running trial:", trial_idx)
        train_split, test_split = get_splits(args, loss, g)
        
        args.use_opt_bound = False

        ## Bound everything that needs to be bounded
        cdf_bounds = get_bounds(args, train_split, test_split)
        ## Compute val objective bounds
        
        best_exp_ind = None
        
        print("Computing val losses...")
        val_losses = dict()
        total_loss = np.zeros(args.num_hypotheses)
        for l_idx, loss_fn in enumerate(loss_fns):
            val_loss = loss_fn.compute(train_split, cdf_bounds, "val")
            if l_idx >= 1:
                val_loss = args.lamb*val_loss
            val_losses[loss_fn.loss_name] = val_loss
            total_loss += val_loss
            
            if loss_fn.loss_name == "Expected":
                best_exp_ind = np.argmin(val_loss)

        best_ind = np.argmin(total_loss)
        best_guar = total_loss[best_ind]
        print("Best ind:", best_ind, " | Best Guar", best_guar)
        for k, v in val_losses.items():
            print(k, v[best_ind])
            
        print("total_loss", total_loss)

        hyp_inds = [-28, -20]
        
        print("total_loss Low", total_loss[hyp_inds[0]])
        print("total_loss High", total_loss[hyp_inds[1]])
        
        hyp_labels = ["Low", "High"]
        
        for i, hyp_ind in enumerate(hyp_inds):
            
            hyp_label = hyp_labels[i]

            plt.rcParams["figure.figsize"] = (6,3)
            g_names = ["Male Over 55", "Other"]

            for j, g_name in enumerate(g_names):

                X_g = test_split.X[
                        torch.where(test_split.g[:,j] == 1)[0]
                ]
                X_g = X_g[:, hyp_ind].numpy()
                X_g = np.sort(X_g)
                X_g = np.cumsum(X_g)/np.sum(X_g)
                ax = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]

                g_label = g_name

                plt.plot(ax, X_g, label=g_label)
            plt.plot([0,1],[0,1], "--", label="Line of equality")
            plt.legend(fontsize=14)
            plt.xlabel(r'$\beta$', fontsize=14)
            plt.ylabel("Cum. Loss Share", fontsize=14)
            plt.title(r'$h_{}$'.format(i), fontsize=16)
            plt.savefig(
                "../plots/{}_{}_{}_{}_{}_test_loss_dist.png".format(
                    args.dataset, 
                    args.metric,
                    args.loss,
                    args.lamb,
                    hyp_label
                ),
                bbox_inches="tight"
            )
            plt.clf()
            
        plt.rcParams["figure.figsize"] = (6,3)
        plt.plot(thresholds, total_loss, "--", color="k", label="Bound")
        plt.scatter(thresholds[hyp_inds[0]], total_loss[hyp_inds[0]], label=r'$h_0$', s=144)
        plt.scatter(thresholds[hyp_inds[1]], total_loss[hyp_inds[1]], label=r'$h_1$', s=144)

        plt.legend(fontsize=14)
        plt.title("Atkinson Index", fontsize=16)
        plt.xlabel("Threshold", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.savefig(
            "../plots/{}_{}_{}_{}_t_vs_b.png".format(
                args.dataset, 
                args.metric,
                args.loss,
                args.lamb,
            ),
            bbox_inches="tight"
        )
        plt.clf()
        
        val_losses["total"] = total_loss
        
        for k, v in val_losses.items():
            rows.append((trial_idx, k, "Guarantee", v[best_ind]))

    df = pd.DataFrame(rows, columns=["Trial", "Loss", "Split", "Value"])
    mean_df = df.groupby(["Loss", "Split"]).mean()["Value"].reset_index()
    
    print("\nMean")
    print(mean_df)
    mean_df.to_csv(
        "../results/{}_{}_{}.csv".format(
            args.dataset, 
            args.metric,
            args.loss
        ),
        float_format="%.4f",
        index=False
    )

    print(mean_df.to_latex(index=False, float_format="%.3f",))


        ## Random Run Tables and Plot

    ## Full Tables and Plot
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run interval experiments")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--no_trials", type=int, default=1, help="no. random trials"
    )
    
    parser.add_argument(
        "--n_val",
        type=int,
        default=1500,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=1000,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--max_full_pop",
        type=int,
        default=1500,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--dataset",
        default="ml-1m",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--metric", default="recall", type=str, help="Inner loss function"
    )
    parser.add_argument(
        "--loss", default="atkinson", type=str, help="Outer loss function"
    )
    parser.add_argument(
        "--group_type", type=str, default=None, help="Group type"
    )
    parser.add_argument(
        "--num_hypotheses", type=int, default=100, help="no. of hypotheses"
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=1.0,
        help="weighting of second loss",
    )

    args = parser.parse_args()
    main(args)
