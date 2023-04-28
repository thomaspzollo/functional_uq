import torch
import numpy as np
from src.bounds import *

# from src.utils import predict
# from src.metric import balanced_accuracy
# from src.metric import fdr

      
class Loss:
    
    def __init__(self):
        pass
    
    def compute_val(self, train_split, bounds):
        pass
    
    def compute_test(self,):
        pass
    

class ExpectedLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Expected"
    
    def compute_val(self, train_split, bounds):
        L_bound = bounds["full"]["L"]
        loss = integrate_quantiles(train_split.X[:self.args.max_full_pop].T, L_bound)
        return loss
    
    def compute_test(self, test_split, best_ind):
        loss = torch.mean(test_split.X[:, best_ind].float())
        return loss.item()
    
    
class CVaRLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "CVaR"
    
    def compute_val(self, train_split, bounds):
        L_bound = bounds["full"]["L"]
        loss = integrate_quantiles(train_split.X[:self.args.max_full_pop].T, L_bound, self.args.beta_min_2, self.args.beta_max_2)
        return loss
    
    def compute_test(self, test_split, best_ind):
        int_x = np.arange(1, test_split.X.shape[0] + 1) / test_split.X.shape[0]
        loss = integrate_quantiles(test_split.X.T, int_x, self.args.beta_min_2, self.args.beta_max_2)[best_ind]
        return loss.item()
    

def get_upper_from_lower(L):
                
    U = np.zeros(L.shape)
    no_i = len(L)
    for i in range(no_i):
        U[i] = 1-L[(no_i-(i+1))]
    return U

    
class MaxDiffLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Max Diff."
    
    def compute_val(self, train_split, bounds):
        
        group_aucs_l, group_aucs_u = [],[]

        for ind, var in enumerate(self.args.interest_vars):
            
            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
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
            
            X_g = X_g[:self.args.max_per_group]
            L_BJ_g = bounds[g_name]["L"]
            print(g_name, X_g.shape, L_BJ_g.shape) 
            
            U_BJ_g = get_upper_from_lower(L_BJ_g)
            aucs_g_l = integrate_quantiles(X_g.T, L_BJ_g)
            aucs_g_u = integrate_quantiles(X_g.T, U_BJ_g)
            group_aucs_l.append(torch.Tensor(aucs_g_l))
            group_aucs_u.append(torch.Tensor(aucs_g_u))

        max_group_diff = torch.zeros(self.args.num_hypotheses)
        for u_idx, gau in enumerate(group_aucs_u):
            for l_idx, gal in enumerate(group_aucs_l):
                if u_idx == l_idx:
                    continue  
                g_diff = torch.abs(gau-gal)
                # print(u_idx, l_idx, gau, gal)
                # print()
                max_group_diff = torch.max(max_group_diff, g_diff)
        loss = max_group_diff.numpy()
        return loss
    
    def compute_test(self, test_split, best_ind):
        group_fdr = 0.0

        max_fdr = 0.0
        min_fdr = 1.0
        for ind, var in enumerate(self.args.interest_vars):
            
            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (test_split.g[:,inds[0]] == 1) & 
                    (test_split.g[:,inds[1]] == 1)
                )
                X_g = test_split.X[g_indices][:, best_ind].float()
                g_name = "_".join(var)
            else:
                X_g = test_split.X[
                    torch.where(test_split.g[:,ind] == 1)[0]
                ][:, best_ind].float()
                g_name = var
                
            group_fdr = torch.mean(X_g)
            print(g_name, group_fdr)
            max_fdr = max(max_fdr, group_fdr)
            min_fdr = min(min_fdr, group_fdr)

        loss = max_fdr - min_fdr
        return loss.item()
    

def calc_gini(X, L, U, beta_min=0.0, beta_max=1.0):

    mean_U = integrate_quantiles(X, U)

    b = L
    dist_max = 1.0
    X_sorted = np.sort(X, axis=-1)
    b_lower = np.concatenate([np.zeros(1), b], -1)
    b_upper = np.concatenate([b, np.ones(1)], -1)
    
    # clip bounds to [beta_min, 1]
    b_lower = np.maximum(b_lower, beta_min)
    b_upper = np.maximum(b_upper, b_lower)
    
    # clip bounds to [0, beta_max]
    b_upper = np.minimum(b_upper, beta_max)
    b_lower = np.minimum(b_upper, b_lower)

    heights = b_upper - b_lower
    widths = np.concatenate([X_sorted, np.full((X_sorted.shape[0], 1), dist_max)], -1)

    res = np.cumsum(heights * widths, -1)/np.expand_dims(mean_U, -1)
    res *= heights
    res = np.sum(res, -1)
    return res

    
class GiniLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Gini"
    
    def compute_val(self, train_split, bounds):
        L = bounds["full"]["L"]
        U = get_upper_from_lower(L)
        loss = calc_gini(train_split.X.T, L, U)
        return loss
    
    def compute_test(self, test_split, best_ind):
        int_x = np.arange(1, test_split.X.shape[0] + 1) / test_split.X.shape[0]
        loss = calc_gini(test_split.X.T, int_x, int_x)[best_ind]
        return loss
    

class ExpectedGroupLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Expected Group"
    
    def compute_val(self, train_split, bounds):
        
        group_aucs = []

        for ind, var in enumerate(self.args.interest_vars):

            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
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
            
            X_g = X_g[:self.args.max_per_group]
            L_BJ_g = bounds[g_name]["L"]

            aucs_g = integrate_quantiles(X_g.T, L_BJ_g, self.args.beta_min_1, self.args.beta_max_1)
            group_aucs.append(torch.Tensor(aucs_g))

        group_aucs = torch.vstack(group_aucs)
        loss = torch.mean(group_aucs, 0).numpy()
        return loss
    
    def compute_test(self, test_split, best_ind):
        
        group_losses = []
        for ind, var in enumerate(self.args.interest_vars):

            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (test_split.g[:,inds[0]] == 1) & 
                    (test_split.g[:,inds[1]] == 1)
                )
                X_g = test_split.X[g_indices]
                g_name = "_".join(var)
            else:
                X_g = test_split.X[
                    torch.where(test_split.g[:,ind] == 1)[0]
                ]
                g_name = var
                
            int_x = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]
            aucs_g = integrate_quantiles(
                X_g.T, int_x, 
                self.args.beta_min_1, self.args.beta_max_1
            )[best_ind]
            group_losses.append(aucs_g.item())

        loss = np.mean(group_losses)
        return loss.item()
    
    
    
class WorstCaseLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Worst Case"
    
    def compute_val(self, train_split, bounds):
        
        group_aucs = []

        for ind, var in enumerate(self.args.interest_vars):

            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
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
            
            X_g = X_g[:self.args.max_per_group]
            L_BJ_g = bounds[g_name]["L"]
            
            aucs_g = integrate_quantiles(X_g.T, L_BJ_g, self.args.beta_min_2, self.args.beta_max_2)
            group_aucs.append(torch.Tensor(aucs_g))

        group_aucs = torch.vstack(group_aucs)
        loss = torch.max(group_aucs, 0)[0].numpy()
        return loss
    
    def compute_test(self, test_split, best_ind):
        
        group_losses = []
        for ind, var in enumerate(self.args.interest_vars):

            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (test_split.g[:,inds[0]] == 1) & 
                    (test_split.g[:,inds[1]] == 1)
                )
                X_g = test_split.X[g_indices]
                g_name = "_".join(var)
            else:
                X_g = test_split.X[
                    torch.where(test_split.g[:,ind] == 1)[0]
                ]
                g_name = var
                
            int_x = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]
            aucs_g = integrate_quantiles(
                X_g.T, int_x, 
                self.args.beta_min_2, self.args.beta_max_2
            )
            aucs_g = aucs_g[best_ind]
            group_losses.append(aucs_g.item())

        loss = np.max(group_losses)
        return loss.item()


def load_losses(args):
    
    losses = []
    
    if "expected" in args.loss:
        losses.append(ExpectedLoss(args))
    if "cvar" in args.loss:
        losses.append(CVaRLoss(args))
    if "exp_group" in args.loss:
        losses.append(ExpectedGroupLoss(args))
    if "interval" in args.loss:
        losses.append(IntervalLoss(args))
    if "gini" in args.loss:
        losses.append(GiniLoss(args))
    if "max_diff" in args.loss:
        losses.append(MaxDiffLoss(args))
    if "worst_case" in args.loss:
        losses.append(WorstCaseLoss(args))
        
    return losses