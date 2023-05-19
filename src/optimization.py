import numpy as np
import scipy
import crossprob
from numpy import linalg as LA
import math
import torch

## weight functions 
def phi_cvar(beta, p):
    if (beta < 1) & (beta > 0):
        if p >= beta:
            res = 1/(1-beta)
        else: 
            res = 0
        return res
    else: 
        print("error: beta not in [0,1]")
        
def phi_mean(p):
    return 1

def phi_p(p, beta_min=0.0, beta_max=1.0, d=2):
    if (beta_max <= 1) & (beta_min >= 0) & (beta_min < beta_max):
        if (p <= beta_max) & (p >= beta_min):
            res = p**d
        else: 
            res = 0
        return  res 
    else:
        print("error: beta's not in [0,1] or beta_max <= beta_min")
        
def phi_intvar(beta_min, beta_max, p):
    if (beta_max < 1) & (beta_min > 0) & (beta_min < beta_max):
        if (p < beta_max) & (p > beta_min):
            res = 1/(beta_max - beta_min)
        else: 
            res = 0
        return  res 
    else:
        print("error: beta's not in [0,1] or beta_max <= beta_min")
        
"""Models for selector implementation."""

import torch.nn as nn


class SeedNet(nn.Module):
    """Implements a feed-forward MLP."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
    ):
        super(SeedNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, 1)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)
        
        
#get the desired gradient for a np.array L for the basic big integral
def basic_grad(L): 
    n = len(L)
    g = np.zeros(n)

    g[0] = -n*crossprob.ecdf2(L[1:n], np.ones(n-1), True)
    g[n-1] = -n*crossprob.ecdf2(L[:n-1], L[n-1]*np.ones(n-1), True)
    for k in range(n-2):
        g[k+1] = -math.factorial(n)/(math.factorial(k+1)*math.factorial(n-k-2))* crossprob.ecdf2(L[:k+1], L[k+1]*np.ones(k+1), True)*crossprob.ecdf2(L[k+2:n], np.ones(n-k-2), True)
    
    return g

#gradient of weight function optimization part for L, X is the input of n+1 dimension, where the last element is the upper bound of X
def weight_grad(phi, L, X): 
    n = len(L)
    g = np.zeros(n)
    for i in range(n):
        g[i] = phi(L[i])*(X[i]-X[i+1])
    
    return g
   

#get the desired gradient for a np.array B for softmax version, B's length is n+1
def softmax_grad_exp(B): 
    n = len(B)
    e_B = np.exp(B)
    L = np.zeros(n)
    G = np.zeros(n*n).reshape(n,n) #paritial L_i/b_j
    
    for i in range(n):
        for j in range(n):
            if j > i:
                G[i][j] = -e_B[j]*np.sum(e_B[:i+1])/(np.sum(e_B)+1)**2
            else:
                G[i][j] = e_B[j]*(1+np.sum(e_B[i+1:]))/(np.sum(e_B)+1)**2
    
    G = np.transpose(G)
    
    L = (np.cumsum(e_B)/(np.sum(e_B)+1)).astype("float64")
    bg = basic_grad(L)
    g = np.sum(np.multiply(bg, G), -1)

    return g


# gradient of loss function, X is the input of n+1 dimension, where the last element is the upper bound of X
def loss_grad_exp(phi, X, B):
    n = len(B)
    e_B = np.exp(B)
    L = np.zeros(n)
    G = np.zeros(n*n).reshape(n,n) #paritial L_i/b_j
    
    for i in range(n):
        for j in range(n):
            if j > i:
                G[i][j] = -e_B[j]*np.sum(e_B[:i+1])/(np.sum(e_B)+1)**2
            else:
                G[i][j] = e_B[j]*(1+np.sum(e_B[i+1:]))/(np.sum(e_B)+1)**2
    
    G = np.transpose(G)
    
    L = (np.cumsum(e_B)/(np.sum(e_B)+1)).astype("float64")
    wg = weight_grad(phi, L, X)    
    g = np.sum(np.multiply(wg, G), -1)

    return g


def quantile_based_loss(X, b, beta_min=0.0, beta_max=1.0, weighted=False, weight_d=2):
    dist_max = 1.0
    b_lower = torch.concat([torch.zeros(1).cuda(), b], -1)
    b_upper = torch.concat([b, torch.ones(1).cuda()], -1)
    
    # clip bounds to [beta_min, 1]
    b_lower = torch.maximum(b_lower, torch.Tensor([beta_min]).cuda())
    b_upper = torch.maximum(b_upper, b_lower)
    
    # clip bounds to [0, beta_max]
    b_upper = torch.minimum(b_upper, torch.Tensor([beta_max]).cuda())
    b_lower = torch.minimum(b_upper, b_lower)
    
    heights = b_upper - b_lower
    widths = torch.concat([X, torch.full((X.shape[0], 1), dist_max).cuda()], -1)
    
    if not weighted:
        res = torch.sum(heights * widths, -1) / (beta_max - beta_min)
    else:
        d = weight_d + 1
        weights = (b_upper**d-b_lower**d)/d
        res = torch.sum(widths * weights, -1) / (beta_max - beta_min)

    return res


def weighted_quantile_based_loss(X, b, beta_min=0.0, beta_max=1.0):
    dist_max = 1.0
    b_lower = torch.concat([torch.zeros(1, requires_grad=True).cuda(), b], -1)
    b_upper = torch.concat([b, torch.ones(1, requires_grad=True).cuda()], -1)
    
    # clip bounds to [beta_min, 1]
    b_lower = torch.maximum(b_lower, torch.Tensor([beta_min]).cuda())
    b_upper = torch.maximum(b_upper, b_lower)
    
    # clip bounds to [0, beta_max]
    b_upper = torch.minimum(b_upper, torch.Tensor([beta_max]).cuda())
    b_lower = torch.minimum(b_upper, b_lower)
    
    heights = b_upper - b_lower
    
    widths = torch.concat([X, torch.full((X.shape[0], 1), dist_max, requires_grad=True).cuda()], -1)
    
    res = torch.sum(heights * widths, -1) / (beta_max - beta_min)

    return res