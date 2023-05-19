import torch
import numpy as np
from src.bounds import *
from src.metric import calc_gini


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
        
    def compute(self, data, bounds, split):
        
        if split == "val":
            ax = bounds["full"]["L"]
            n_data = self.args.max_full_pop
        else:
            ax = np.arange(1, data.X.shape[0] + 1) / data.X.shape[0]
            n_data = data.X.shape[0]
        loss = integrate_quantiles(data.X[:n_data].T, ax)
        return loss


class CVaRLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "CVaR"
        
    def compute(self, data, bounds, split):
        
        if split == "val":
            ax = bounds["full"]["L"]
            n_data = self.args.max_full_pop
        else:
            ax = np.arange(1, data.X.shape[0] + 1) / data.X.shape[0]
            n_data = data.X.shape[0]
            
        loss = integrate_quantiles(
            data.X[:n_data].T, 
            ax, 
            self.args.beta_min_2, 
            self.args.beta_max_2
        )
        return loss

    
class ExpectedGroupLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Expected Group"
        
    def compute(self, data, bounds, split):
        
        group_aucs = []

        for ind, var in enumerate(self.args.interest_vars):

            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (data.g[:,inds[0]] == 1) & 
                    (data.g[:,inds[1]] == 1)
                )
                X_g = data.X[g_indices]
                g_name = "_".join(var)
                
            else:
                X_g = data.X[
                    torch.where(data.g[:,ind] == 1)[0]
                ]
                g_name = var
                  
            if split == "val":
                X_g = X_g[:self.args.max_per_group]
                if self.args.use_opt_bound:
                    ax = bounds[g_name]["L_opt"]
                elif self.args.use_dkw_bound:
                    ax = bounds[g_name]["L_dkw"]
                else:
                    ax = bounds[g_name]["L"]
            else:
                ax = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]

            aucs_g = integrate_quantiles(X_g.T, ax, self.args.beta_min_1, self.args.beta_max_1)
            group_aucs.append(torch.Tensor(aucs_g))
            
        group_aucs = torch.vstack(group_aucs)
        loss = torch.mean(group_aucs, 0).numpy()
        return loss

    
class WorstCaseLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Worst Case"
        
        
    def compute(self, data, bounds, split):
        
        group_aucs = []

        for ind, var in enumerate(self.args.interest_vars):

            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (data.g[:,inds[0]] == 1) & 
                    (data.g[:,inds[1]] == 1)
                )
                X_g = data.X[g_indices]
                g_name = "_".join(var)
            else:
                X_g = data.X[
                    torch.where(data.g[:,ind] == 1)[0]
                ]
                g_name = var
                
            if split == "val":
                X_g = X_g[:self.args.max_per_group]
                ax = bounds[g_name]["L"]
            else:
                ax = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]

            aucs_g = integrate_quantiles(X_g.T, ax, self.args.beta_min_2, self.args.beta_max_2)
            group_aucs.append(torch.Tensor(aucs_g))

        group_aucs = torch.vstack(group_aucs)
        loss = torch.max(group_aucs, 0)[0].numpy()
        return loss

    
class MaxDiffLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Max Diff."
        
        
    def compute(self, data, bounds, split):
        # if split == "val":
        #     loss = self.compute_val(data, bounds)
        # else:
        #     loss = self.compute_test(data)
        # return loss
        
        if split == "val":
            group_aucs_l, group_aucs_u = [],[]
        if split == "test":
            group_aucs = []

        for ind, var in enumerate(self.args.interest_vars):
            
            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (data.g[:,inds[0]] == 1) & 
                    (data.g[:,inds[1]] == 1)
                )
                X_g = data.X[g_indices]
                g_name = "_".join(var)
            else:
                X_g = data.X[
                    torch.where(data.g[:,ind] == 1)[0]
                ]
                g_name = var
            
            if split == "val":
                
                X_g = X_g[:self.args.max_per_group]
                if self.args.use_opt_bound:
                    L_BJ_g = bounds[g_name]["L_opt"]
                else:
                    L_BJ_g = bounds[g_name]["L"]
                U_BJ_g = get_upper_from_lower(L_BJ_g)
                aucs_g_l = integrate_quantiles(X_g.T, L_BJ_g, self.args.beta_min_2, self.args.beta_max_2)
                aucs_g_u = integrate_quantiles_upper(X_g.T, U_BJ_g, self.args.beta_min_2, self.args.beta_max_2)
                group_aucs_l.append(torch.Tensor(aucs_g_l))
                group_aucs_u.append(torch.Tensor(aucs_g_u))
            
            else:
                
                ax = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]
                aucs_g = integrate_quantiles(X_g.T, ax, self.args.beta_min_2, self.args.beta_max_2)
                group_aucs.append(aucs_g)

        if split == "val":
            max_group_diff = torch.zeros(self.args.num_hypotheses)
            for u_idx, gau in enumerate(group_aucs_u):
                for l_idx, gal in enumerate(group_aucs_l):
                    if u_idx == l_idx:
                        continue  
                    g_diff_1 = gau-gal
                    g_diff_2 = gal-gau
                    g_diff = torch.max(g_diff_1, g_diff_2)
                    max_group_diff = torch.max(max_group_diff, g_diff)
            loss = max_group_diff.numpy()
            # print("val loss", loss.shape)
        if split == "test":
            group_aucs = np.vstack(group_aucs)
            loss = (np.max(group_aucs, 0) - np.min(group_aucs, 0))
            # print("group aucs", group_aucs.shape)
            # print("test loss", loss.shape)
        return loss

    
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
            U_BJ_g = get_upper_from_lower(L_BJ_g)
            aucs_g_l = integrate_quantiles(X_g.T, L_BJ_g)
            aucs_g_u = integrate_quantiles_upper(X_g.T, U_BJ_g)
            group_aucs_l.append(torch.Tensor(aucs_g_l))
            group_aucs_u.append(torch.Tensor(aucs_g_u))

        max_group_diff = torch.zeros(self.args.num_hypotheses)
        for u_idx, gau in enumerate(group_aucs_u):
            for l_idx, gal in enumerate(group_aucs_l):
                if u_idx == l_idx:
                    continue  
                g_diff = torch.abs(gau-gal)
                max_group_diff = torch.max(max_group_diff, g_diff)
        loss = max_group_diff.numpy()
        return loss
    
    def compute_test(self, test_split):
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
                X_g = test_split.X[g_indices].float()
                g_name = "_".join(var)
            else:

                X_g = test_split.X[
                    torch.where(test_split.g[:,ind] == 1)[0]
                ].float()
                              
                g_name = var
                
            group_fdr = torch.mean(X_g, 0)
            
            max_fdr = np.maximum(max_fdr, group_fdr)
            min_fdr = np.minimum(min_fdr, group_fdr)

        loss = max_fdr - min_fdr
        return loss.numpy()
    
    
class WorstCaseDeltaLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Worst Case Delta"
        
        
    def compute(self, data, bounds, split):
        
        group_aucs = []

        for ind, var in enumerate(self.args.interest_vars):

            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (data.g[:,inds[0]] == 1) & 
                    (data.g[:,inds[1]] == 1)
                )
                X_g = data.X[g_indices]
                g_name = "_".join(var)
            else:
                X_g = data.X[
                    torch.where(data.g[:,ind] == 1)[0]
                ]
                g_name = var
                
            if split == "val":
                X_g = X_g[:self.args.max_per_group]
                ax = bounds[g_name]["L"]
            else:
                ax = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]

            aucs_g = integrate_smooth_delta(X_g.T, ax)
            group_aucs.append(torch.Tensor(aucs_g))

        
        group_aucs = torch.vstack(group_aucs)
        loss = torch.max(group_aucs, 0)[0].numpy()
        return loss

    
class MaxDiffDeltaLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Max Diff. Delta"
        
        
    def compute(self, data, bounds, split):
        # if split == "val":
        #     loss = self.compute_val(data, bounds)
        # else:
        #     loss = self.compute_test(data)
        # return loss
        
        if split == "val":
            group_aucs_l, group_aucs_u = [],[]
        else:
            group_aucs = []
            
        for ind, var in enumerate(self.args.interest_vars):
            
            if self.args.group_type == "intersectional":
                inds = []
                for g_n in var:
                    inds.append(self.args.var_key.index(g_n))
                g_indices = torch.where(
                    (data.g[:,inds[0]] == 1) & 
                    (data.g[:,inds[1]] == 1)
                )
                X_g = data.X[g_indices]
                g_name = "_".join(var)
            else:
                X_g = data.X[
                    torch.where(data.g[:,ind] == 1)[0]
                ]
                g_name = var
            
            if split == "val":
                X_g = X_g[:self.args.max_per_group]
                if self.args.use_opt_bound:
                    L_BJ_g = bounds[g_name]["L_opt"]
                elif self.args.use_dkw_bound:
                    L_BJ_g = bounds[g_name]["L_dkw"]
                else:
                    L_BJ_g = bounds[g_name]["L"]
                U_BJ_g = get_upper_from_lower(L_BJ_g)
                aucs_g_l = integrate_smooth_delta(X_g.T, L_BJ_g)
                aucs_g_u = integrate_smooth_delta_upper(X_g.T, U_BJ_g)
                group_aucs_l.append(torch.Tensor(aucs_g_l))
                group_aucs_u.append(torch.Tensor(aucs_g_u))
            else:
                ax = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]
                aucs = integrate_smooth_delta(X_g.T, ax)
                group_aucs.append(torch.Tensor(aucs))

        if split == "val":
            max_group_diff = torch.zeros(self.args.num_hypotheses)
            for u_idx, gau in enumerate(group_aucs_u):
                for l_idx, gal in enumerate(group_aucs_l):
                    if u_idx == l_idx:
                        continue  
                    # g_diff = torch.abs(gau-gal)
                    # max_group_diff = torch.max(max_group_diff, g_diff)
                    g_diff_1 = gau-gal
                    g_diff_2 = gal-gau
                    g_diff = torch.max(g_diff_1, g_diff_2)
                    max_group_diff = torch.max(max_group_diff, g_diff)
            loss = max_group_diff.numpy()
        else:
            group_aucs = np.vstack(group_aucs)
            loss = (np.max(group_aucs, 0) - np.min(group_aucs, 0))
        
        return loss
            
                

    
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
            U_BJ_g = get_upper_from_lower(L_BJ_g)
            aucs_g_l = integrate_smooth_delta(X_g.T, L_BJ_g)
            aucs_g_u = integrate_smooth_delta_upper(X_g.T, U_BJ_g)
            group_aucs_l.append(torch.Tensor(aucs_g_l))
            group_aucs_u.append(torch.Tensor(aucs_g_u))

        max_group_diff = torch.zeros(self.args.num_hypotheses)
        for u_idx, gau in enumerate(group_aucs_u):
            for l_idx, gal in enumerate(group_aucs_l):
                if u_idx == l_idx:
                    continue  
                g_diff = torch.abs(gau-gal)
                max_group_diff = torch.max(max_group_diff, g_diff)
        loss = max_group_diff.numpy()
        return loss
    
    def compute_test(self, test_split):
        
        group_aucs = []
        
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
            ax = np.arange(1, X_g.shape[0] + 1) / X_g.shape[0]
            aucs = integrate_smooth_delta(X_g.T, ax)
            group_aucs.append(torch.Tensor(aucs))
            
        group_aucs = torch.vstack(group_aucs)
        max_loss = torch.max(group_aucs, 0)[0].numpy()
        min_loss = torch.min(group_aucs, 0)[0].numpy()
        loss = max_loss - min_loss
        return loss
    
    
def calc_atkinson(X, L, args, beta_min=0.0, beta_max=1.0):
    
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
    
    widths = np.concatenate([X_sorted, np.full((X_sorted.shape[0], 1), dist_max)], -1)
    
    num = (X_sorted**0.5) * ( np.flip(b_lower[1:]) - np.flip(b_lower[:-1]) )
    num = np.sum(num, -1)**2
    den = np.sum((b_upper-b_lower)*widths, -1)
    res = 1-(num/den)
    res = np.minimum(res, np.ones_like(res))
    res = np.maximum(res, np.zeros_like(res))
    
    return res

    
class AtkinsonLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Atkinson"
        
        
    def compute(self, data, bounds, split):

        if split == "val":
            L = bounds["full"]["L"]
            n_data = self.args.max_full_pop
            loss = calc_atkinson(data.X.T[:n_data], L, self.args)
        else:
            L = np.arange(1, data.X.shape[0] + 1) / data.X.shape[0]
            n_data = data.X.shape[0]
            loss = calc_atkinson(data.X.T[:n_data], L, self.args)

        return loss


import statsmodels.api as sm
def calc_gini(X, L, args, beta_min=0.0, beta_max=1.0):

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
    
    # heights = b_upper - b_lower
    widths = np.concatenate([X_sorted, np.full((X_sorted.shape[0], 1), dist_max)], -1)
    
    num = np.sum((b_upper**2-b_lower**2)*widths, -1)
    den = np.sum((np.flip(b_lower[1:])-np.flip(b_lower[:-1]))*X_sorted, -1)
    res = (num/den)-1.0
    res = np.minimum(res, np.ones_like(res))
    return res

    
class GiniLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Gini"
        
        
    def compute(self, data, bounds, split):

        if split == "val":
            L = bounds["full"]["L"]
            n_data = self.args.max_full_pop
            loss = calc_gini(data.X.T[:n_data], L, self.args)
        else:
            L = np.arange(1, data.X.shape[0] + 1) / data.X.shape[0]
            n_data = data.X.shape[0]
            loss = calc_gini(data.X.T[:n_data], L, self.args)

        return loss

    
class DeltaLoss(Loss):
    
    def __init__(self, args=None):
        self.args = args
        self.loss_name = "Expected"
        
    def compute(self, data, bounds, split):
        
        if split == "val":
            ax = bounds["full"]["L"]
            n_data = self.args.max_full_pop
        else:
            ax = np.arange(1, data.X.shape[0] + 1) / data.X.shape[0]
            n_data = data.X.shape[0]
        loss = integrate_smooth_delta(data.X[:n_data].T, ax)
        return loss


def load_losses(args):
    
    losses = []
    
    if "expected" in args.loss:
        losses.append(ExpectedLoss(args))
    if "cvar" in args.loss:
        losses.append(CVaRLoss(args))
    if "exp_group" in args.loss:
        losses.append(ExpectedGroupLoss(args))
    if "gini" in args.loss:
        losses.append(GiniLoss(args))
    if "atkinson" in args.loss:
        losses.append(AtkinsonLoss(args))
    if "max_diff" in args.loss:
        losses.append(MaxDiffLoss(args))
    if "worst_case" in args.loss:
        losses.append(WorstCaseLoss(args))
    if "max_delta_diff" in args.loss:
        losses.append(MaxDiffDeltaLoss(args))
    if "worst_delta_case" in args.loss:
        losses.append(WorstCaseDeltaLoss(args))
        
    return losses