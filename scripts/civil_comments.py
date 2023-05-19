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

import matplotlib.pyplot as plt


def main(args):
    
    set_all_seeds(args.seed)
    
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
        args.use_dkw_bound = False

        ## Bound everything that needs to be bounded
        cdf_bounds = get_bounds(args, train_split, test_split)
        ## Compute val objective bounds
        print("Computing val losses...")
        val_losses = dict()
        total_loss = np.zeros(args.num_hypotheses)
        for l_idx, loss_fn in enumerate(loss_fns):
            val_loss = loss_fn.compute(train_split, cdf_bounds, "val")
            if l_idx >= 1:
                val_loss = args.lamb*val_loss
            val_losses[loss_fn.loss_name] = val_loss
            total_loss += val_loss
 
        best_ind = np.argmin(total_loss)
        best_guar = total_loss[best_ind]
        print("Best ind:", best_ind, " | Best Guar", best_guar)
        for k, v in val_losses.items():
            print(k, v[best_ind])
            
        plot_losses(val_losses, best_ind, thresholds, total_loss, args, "val")

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

        plot_losses(test_losses, emp_min_ind, thresholds, total_test_loss, args, "test")
        
        if trial_idx == 0:
        
            # X_fixed_pred = train_split.X[:, emp_min_ind]
            X_fixed_pred = train_split.X[:, best_ind]
            # print(X_fixed_pred.shape, train_split.g.shape)

            for ind, var in enumerate(args.interest_vars):

                if args.group_type == "intersectional":
                    inds = []
                    for g_n in var:
                        inds.append(args.var_key.index(g_n))
                    g_indices = torch.where(
                        (train_split.g[:,inds[0]] == 1) & 
                        (train_split.g[:,inds[1]] == 1)
                    )
                    X_g = X_fixed_pred[g_indices]
                    g_name = "_".join(var)

                else:
                    X_g = X_fixed_pred[
                        torch.where(train_split.g[:,ind] == 1)[0]
                    ]
                    g_name = var

                # X_g = X_g[:args.max_per_group]

                bound_save_root = "../data/civil_comments/bound_samples/{}_{}.pt".format(g_name, args.max_per_group)
                torch.save(X_g, bound_save_root)


        total_test_loss = total_test_loss[best_ind]

        val_losses["Total"] = total_loss
        test_losses["Total"] = total_test_loss
        
        for k, v in val_losses.items():
            rows.append((trial_idx, k, "BJ Guarantee", v[best_ind]))
        for k, v in test_losses.items():
            if type(v) not in [np.float32, np.float64]:
                v = v[best_ind]
            rows.append((trial_idx, k, "Test", v))

            
        args.use_opt_bound = True
        for ind, var in enumerate(args.interest_vars):
            
            try:
            
                if args.group_type == "intersectional":
                    g_name = "_".join(var)
                    bound_save_root = "../data/civil_comments/opt_bounds/{}_{}.pt".format(
                        g_name, 
                        args.max_per_group
                    )
                    g_bound = torch.load(bound_save_root).numpy()
                    cdf_bounds[g_name]["L_opt"] = g_bound
                else:
                    raise ValueError
                
            except:
                
                print("Run optimization notebook to include optimized bound")
                
        print("Computing optimized losses...")
        opt_losses = dict()
        total_opt_loss = np.zeros(args.num_hypotheses)
        for l_idx, loss_fn in enumerate(loss_fns):
            opt_loss = loss_fn.compute(train_split, cdf_bounds, "val")
            if l_idx >= 1:
                opt_loss = args.lamb*opt_loss
            opt_losses[loss_fn.loss_name] = opt_loss
            total_opt_loss += opt_loss
            
        opt_losses["Total"] = total_opt_loss
        for k, v in opt_losses.items():
            rows.append((trial_idx, k, "Opt. Guarantee", v[best_ind]))
            
        args.use_opt_bound = False
        args.use_dkw_bound = True
        
        print("Making DKW baseline")
        X_fixed_pred = train_split.X[:, best_ind]

        for ind, var in enumerate(args.interest_vars):

            if args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(args.var_key.index(g_n))
                g_indices = torch.where(
                    (train_split.g[:,inds[0]] == 1) & 
                    (train_split.g[:,inds[1]] == 1)
                )
                X_g = X_fixed_pred[g_indices]
                X_g = X_g[:args.max_per_group]
                
                g_name = "_".join(var)
                
                
                b_lower = np.zeros(args.max_per_group)
                for j in range(args.max_per_group):
                    n_lower = (X_g < ((j+1)/args.max_per_group)).sum()
                    b_lower[j] = n_lower/args.max_per_group - np.sqrt(
                        np.log(2/(args.correction/args.max_per_group))/(2*args.max_per_group)
                    )
                b_lower = np.maximum(b_lower,0.0)
                cdf_bounds[g_name]["L_dkw"] = b_lower
            else:
                raise ValueError
        
        print("Computing DKW losses...")
        dkw_losses = dict()
        total_dkw_loss = np.zeros(args.num_hypotheses)
        for l_idx, loss_fn in enumerate(loss_fns):
            dkw_loss = loss_fn.compute(train_split, cdf_bounds, "val")
            if l_idx >= 1:
                dkw_loss = args.lamb*dkw_loss
            dkw_losses[loss_fn.loss_name] = dkw_loss
            total_dkw_loss += dkw_loss

        dkw_losses["Total"] = total_dkw_loss
        for k, v in dkw_losses.items():
            print(k, v.shape)
            rows.append((trial_idx, k, "DKW Guarantee", v[best_ind]))
            
    df = pd.DataFrame(rows, columns=["Trial", "Loss", "Split", "Value"])
    mean_df = df.groupby(["Loss", "Split"]).mean()["Value"].reset_index()
    
    print("\nMean")
    print(mean_df)
    mean_df.to_csv(
        "../results/{}_{}_{}_{}.csv".format(
            args.dataset, 
            args.metric,
            args.loss,
            args.max_per_group
        ),
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
    
    print(mean_df.to_latex(index=False, float_format="%.5f"))


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
        default=169038,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=200,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--max_full_pop",
        type=int,
        default=2000,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--dataset",
        default="civilcomments",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--metric", default="brier", type=str, help="Inner loss function"
    )
    parser.add_argument(
        "--loss", default="exp_group_max_delta_diff", type=str, help="Outer loss function"
    )
    parser.add_argument(
        "--group_type", type=str, default="intersectional", help="Group type"
    )
    parser.add_argument(
        "--num_hypotheses", type=int, default=50, help="no. of hypotheses"
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=1.0,
        help="weighting of second loss",
    )

    args = parser.parse_args()
    main(args)
