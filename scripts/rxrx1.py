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
    
    Z, X, y, g = load_dataset(args)
    
    if args.dataset == "rxrx1":
        print("Z, X, y:", Z.shape, X.shape, y.shape)
    if args.dataset == "civilcomments":    
        print("g, X, y:", g.shape, X.shape, y.shape)

    thresholds = get_thresholds(args)

    preds = get_preds(args, X, thresholds)        
    loss = get_loss(args, preds, y)
    assert (loss.shape[0] == X.shape[0]) and (loss.shape[1] == args.num_hypotheses)
    args.var_key, args.interest_vars, args.no_groups = load_group_info(args)    

    load_configs(args)
    
    args.correction = args.delta/(args.num_hypotheses*args.no_bounds)
    
    args.split_interval = (args.beta_max_2 is not None)
    
    loss_fns = load_losses(args)

    print("t", thresholds.shape, "preds", preds.shape, "loss shape", loss.shape)
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
        
        plt.rcParams["figure.figsize"] = (6,2)
        X_best = np.sort(test_split.X[:,best_ind])
        X_exp = np.sort(test_split.X[:,best_exp_ind])
        ax = np.arange(1, X_best.shape[0] + 1) / X_best.shape[0]
        plt.plot(ax, np.cumsum(X_exp)/np.sum(X_exp), label="Expected")
        plt.plot(ax, np.cumsum(X_best)/np.sum(X_best), label="Expected+Gini")
        plt.plot([0,1], [0,1], "--", label="Line of Equality")
        plt.legend()
        plt.xlabel(r'$\beta$', fontsize=14)
        plt.ylabel("Cum. Loss Share", fontsize=14)
        plt.savefig(
            "../plots/{}_{}_{}_test_loss_dist.png".format(
                args.dataset, 
                args.metric,
                args.loss,
            ),
            bbox_inches="tight"
        )
        plt.clf()
        
        plot_losses_single(val_losses, best_ind, thresholds, total_loss, args, "val")

        print()
        print("Computing test losses...")
        ## Compute test objective quantities
        test_losses = dict()
        total_test_loss = 0.0
        for l_idx, loss_fn in enumerate(loss_fns):
            test_loss = loss_fn.compute(test_split, None, "test")
            if l_idx >= 1:
                test_loss = args.lamb*test_loss
            test_losses[loss_fn.loss_name] = test_loss
            total_test_loss += test_loss
            print(loss_fn.loss_name, test_loss[best_ind])
            
        emp_min_ind = np.argmin(total_test_loss)

        # plot_losses(test_losses, emp_min_ind, thresholds, total_test_loss, args, "test")

        total_test_loss = total_test_loss[best_ind]

        val_losses["total"] = total_loss
        test_losses["total"] = total_test_loss
        
        for k, v in val_losses.items():
            rows.append((trial_idx, k, "Guarantee", v[best_ind]))
        for k, v in test_losses.items():
            if type(v) not in [np.float32, np.float64]:
                v = v[best_ind]
            rows.append((trial_idx, k, "Actual", v))
  
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
    
#     print("\nStdev.")
#     std_df = df.groupby(["Loss", "Split"]).std()["Value"].reset_index()
#     print(std_df)
#     std_df.to_csv(
#         "../results/{}_{}_{}_std.csv".format(
#             args.dataset, 
#             args.metric,
#             args.loss
#         ),
#         float_format="%.4f",
#         index=False
#     )
    
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
        default=2500,
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
        default=2500,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--dataset",
        default="rxrx1",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--metric", default="balanced_acc", type=str, help="Inner loss function"
    )
    parser.add_argument(
        "--loss", default="expected_gini", type=str, help="Outer loss function"
    )
    parser.add_argument(
        "--group_type", type=str, default=None, help="Group type"
    )
    parser.add_argument(
        "--num_hypotheses", type=int, default=50, help="no. of hypotheses"
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=0.2,
        help="weighting of second loss",
    )

    args = parser.parse_args()
    main(args)
