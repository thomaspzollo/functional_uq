import argparse
from argparse import Namespace
from dataclasses import dataclass
import os
import random

import torch
import numpy as np
from sklearn.metrics import brier_score_loss

from src.bounds import *
from src.utils import predict
from src.metric import balanced_accuracy
from src.metric import fdr
from src.loss_fns import *

import matplotlib.pyplot as plt


def load_dataset(args):
    
    Z, X, y, g = None, None, None, None
    
    if args.dataset == "rxrx1":
        
        save_root = "../data/rxrx1/"
        Z = torch.load(save_root+"rxrx_val_logits.pt")
        X = torch.softmax(Z, -1)
        y = torch.load(save_root+"rxrx_val_labels.pt").int()
        
    elif args.dataset == "civilcomments":
        
        save_root = "../data/civil_comments/"
        X = torch.load(save_root+"X_train.pt")
        y = torch.load(save_root+"y_train.pt")
        g = torch.load(save_root+"g_train.pt")
        
    else:
        raise ValueError
        
    return Z, X, y, g
        

def load_group_info(args):

    if args.group_type:

        if args.dataset == "civilcomments":
            if args.group_type == "top_level":
                var_key = None
                interest_vars = ["male", "female", "black", "white"]
            elif args.group_type == "intersectional":
                var_key = ["male", "female", "black", "white"]
                interest_vars = [
                    ("male","black"),("female","black"),
                    ("male","white"),("female","white")
                ]
        else:
            raise ValueError
        no_groups = len(interest_vars)

    else:
        var_key, interest_vars, no_groups = None, None, 1

    return var_key, interest_vars, no_groups

    
def get_preds(args, X, thresholds):

    if args.metric == "brier":
        preds = calibrator(X, thresholds)
    else:
        preds = torch.gt(X.unsqueeze(-1), thresholds).int()
    return preds


def get_thresholds(args):

    if args.dataset == "rxrx1":
        if args.metric == "balanced_acc":
            thresholds = torch.logspace(-10, 0, args.num_hypotheses)
        else:
            raise ValueError
    elif args.dataset == "civilcomments":
        if args.metric == "brier":
            thresholds = torch.linspace(0.1, 3.0, args.num_hypotheses)
        elif args.metric == "fdr":
            thresholds = torch.linspace(0.00, 1.0, args.num_hypotheses)
        else:
            raise ValueError
    return thresholds


def get_loss(args, preds, y):

    if args.metric == "brier":
        loss = torch.zeros(preds.shape)
        for i in range(preds.shape[-1]):
            p_i = torch.Tensor(preds[:,i])
            score = (p_i-y)**2
            loss[:,i] = score

    elif args.metric == "fdr":
        loss = fdr(preds.T, y).reshape(-1, args.num_hypotheses)
    elif args.metric == "balanced_acc":
        y_one_hot = torch.zeros((y.shape[0], 1139)).int()
        for i in range(y.shape[0]):
            y_one_hot[i, y[i]]=1
        loss = 1 - balanced_accuracy(
            preds.permute(0,2,1), 
            y_one_hot.unsqueeze(1)
        )
    else: 
        raise ValueError

    return loss


def get_splits(args, loss, g):
    
    p = torch.randperm(loss.size()[0])
    loss = loss[p,:]
    if g is not None:
        g = g[p,:]
        train_split = Split(
            X=loss[:args.n_val],
            g=g[:args.n_val]
        )

        test_split = Split(
            X=loss[args.n_val:],
            g=g[args.n_val:]
        )

    else:
        train_split = Split(
            X=loss[:args.n_val],
            g=None
        )

        test_split = Split(
            X=loss[args.n_val:],
            g=None
        )

    return train_split, test_split


def get_bj_bound(no_ex, corr, beta_min, beta_max):

    bound_save_root = "../bounds/bj/{}_{}_{}_{}.npy".format(
        no_ex, corr, beta_min, beta_max
    )
    try:
        bj_bound = np.load(bound_save_root)
    except:
        bj_bound = berk_jones_two_sided(
            no_ex, 
            corr, 
            beta_min, 
            beta_max
        )
        np.save(bound_save_root, bj_bound)
    return bj_bound


def get_bounds(args, train_split, test_split):
    
    cdf_bounds = {}
    if args.group_type is not None:
        for ind, var in enumerate(args.interest_vars):
            if args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(args.var_key.index(g_n))
                g_indices = torch.where(
                    (train_split.g[:,inds[0]] == 1) & 
                    (train_split.g[:,inds[1]] == 1)
                )
                X_g = train_split.X[g_indices]
                g_name = "_".join(var)
            else:
                X_g = train_split.X[
                    torch.where(train_split.g[:,ind] == 1)[0]
                ]
                g_name = var
                
            X_g = X_g[:args.max_per_group]
            L_BJ_g = get_bj_bound(
                X_g.shape[0], 
                args.correction, 
                0.0, 
                1.0
            )
            cdf_bounds[g_name] = dict()
            cdf_bounds[g_name]["L"] = L_BJ_g

    if (args.group_type is None) or (args.no_groups < args.no_bounds):
        cdf_bounds["full"] = dict()

        L_BJ = get_bj_bound(
            min(args.n_val, args.max_full_pop),
            args.correction, 
            0.0, 
            1.0
        )
        cdf_bounds["full"]["L"] = L_BJ

    return cdf_bounds  


def load_configs(args):

    args.beta_min_2 = None
    args.beta_max_2 = None

    if args.dataset == "civilcomments":
        if args.metric == "brier" or args.loss == "exp_group_worst_case":
            args.no_bounds = 4

            args.beta_min_1 = 0.45
            args.beta_max_1 = 0.55
            args.beta_min_2 = 0.75
            args.beta_max_2 = 1.0

        elif args.metric == "fdr":
            args.no_bounds = 5

            args.beta_min_1 = 0.0
            args.beta_max_1 = 1.0
            args.beta_min_2 = 0.5
            args.beta_max_2 = 1.0
        else:
            raise ValueError
        args.upper_bounds = False
    elif args.dataset == "rxrx1":
        if args.metric == "balanced_acc":

            args.no_bounds = 1
            args.upper_bounds = True

            args.bound_beta_min_1 = 0.0
            args.bound_beta_max_1 = 1.0
            args.loss_beta_min_1 = 0.0
            args.loss_beta_max_1 = 1.0

        else:
            raise ValueError
    else:
        raise ValueError
        

def main(args):
    
    set_all_seeds(args.seed)
    
    Z, X, y, g = load_dataset(args)
    
    if args.dataset == "rxrx1":
        print("Z, X, y:", Z.shape, X.shape, y.shape)
    if args.dataset == "civilcomments":    
        print("g, X, y:", g.shape, X.shape, y.shape)
    
    thresholds = get_thresholds(args)
    print("t", thresholds.shape)
    preds = get_preds(args, X, thresholds)        
    print("preds", preds.shape)
    loss = get_loss(args, preds, y)
    print("loss shape", loss.shape) 
    assert (loss.shape[0] == X.shape[0]) and (loss.shape[1] == args.num_hypotheses)
    args.var_key, args.interest_vars, args.no_groups = load_group_info(args)    
    print("group stuff", args.var_key, args.interest_vars, args.no_groups)
    load_configs(args)
    
    args.correction = args.delta/(args.num_hypotheses*args.no_bounds)
    args.split_interval = (args.beta_max_2 is not None)
    
    loss_fns = load_losses(args)

    print("loss fns", loss_fns)
    print("args", args)
    
    rows = []
    
    for trial_idx in range(args.no_trials):
        
        print()
        print("Trial:", trial_idx)
        print("splits...")
        train_split, test_split = get_splits(args, loss, g)

        ## Bound everything that needs to be bounded

        cdf_bounds = get_bounds(args, train_split, test_split)
        print("cdf_bounds", list(cdf_bounds.keys()))
        print()

        ## Compute val objective bounds
        print("Computing val losses...")
        val_losses = dict()
        total_loss = np.zeros(args.num_hypotheses)
        for l_idx, loss_fn in enumerate(loss_fns):
            val_loss = loss_fn.compute_val(train_split, cdf_bounds)
            if l_idx >= 1:
                val_loss = args.lamb*val_loss
            val_losses[loss_fn.loss_name] = val_loss
            total_loss += val_loss
            
            print(loss_fn.loss_name, val_loss.shape)
            print()

            
        best_ind = np.argmin(total_loss)
        best_guar = total_loss[best_ind]
        print("Best ind:", best_ind, " | Best Guar", best_guar)
        for k, v in val_losses.items():
            print(k, v[best_ind])

        plt.rcParams["figure.figsize"] = (15,3)
        fig, ax = plt.subplots(1,1+len(val_losses)) 
        
        ax_idx = 0
        for k, v in val_losses.items():
            ax[ax_idx].plot(thresholds, v, "--", label=k)
            ax[ax_idx].scatter(thresholds[best_ind], v[best_ind], color="orange", zorder=100)
            ax[ax_idx].set_xlabel("Threshold")
            ax[ax_idx].set_ylabel("Loss Guarantee")
            ax[ax_idx].set_title(k)
            
            if args.dataset == "rxrx1":
                ax[ax_idx].set_xscale("log")
            else:
                ax[ax_idx].set_xscale("linear")
            
            ax_idx += 1

        ax[len(val_losses)].scatter(thresholds, total_loss)
        ax[len(val_losses)].scatter(thresholds[best_ind], total_loss[best_ind], color="orange")
        ax[len(val_losses)].set_xlabel("Threshold")
        ax[len(val_losses)].set_title("Total")
        
        if args.dataset == "rxrx1":
            ax[len(val_losses)].set_xscale("log")
        else:
            ax[len(val_losses)].set_xscale("linear")
        
        plt.savefig(
            "../plots/{}_{}_{}.png".format(
                args.dataset, 
                args.metric,
                args.loss
            ),
            bbox_inches="tight"
        )
        plt.clf()

        print()
        print("Computing test losses...")
        ## Compute test objective quantities
        test_losses = dict()
        total_test_loss = 0.0
        for l_idx, loss_fn in enumerate(loss_fns):
            test_loss = loss_fn.compute_test(test_split, best_ind)
            if l_idx >= 1:
                test_loss = args.lamb*test_loss
            test_losses[loss_fn.loss_name] = test_loss
            total_test_loss += test_loss
            print(loss_fn.loss_name, test_loss)
            
        print("total test loss", total_test_loss)
        
        val_losses["total"] = total_loss
        test_losses["total"] = total_test_loss
        

        for k, v in val_losses.items():
            rows.append((trial_idx, k, "val", v[best_ind]))
        for k, v in test_losses.items():
            rows.append((trial_idx, k, "test", v))
            
    df = pd.DataFrame(rows, columns=["Trial", "Loss", "Split", "Value"])
    df = df.groupby(["Loss", "Split"]).mean()["Value"].reset_index()
    print(df)
    df.to_csv(
        "../results/{}_{}_{}.csv".format(
            args.dataset, 
            args.metric,
            args.loss
        ),
        float_format="%.4f",
        index=False
    )
    print(df.to_latex(index=False))


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
        "--no_trials", type=int, default=10, help="no. random trials"
    )
    
    parser.add_argument(
        "--n_val",
        type=int,
        default=1000,
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
        default=1000,
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
        "--loss", default="expected", type=str, help="Outer loss function"
    )
    parser.add_argument(
        "--group_type", type=str, default=None, help="Group type"
    )
    parser.add_argument(
        "--num_hypotheses", type=int, default=200, help="no. of hypotheses"
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=1.0,
        help="weighting of second loss",
    )

    args = parser.parse_args()
    main(args)
