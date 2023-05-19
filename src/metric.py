import torch
from torch import Tensor
from src.bounds import integrate_quantiles, integrate_quantiles_upper
import numpy as np

def sensitivity(input: Tensor, target: Tensor) -> Tensor:
    """Compute the sensitivity (recall).

    Args:
        input (Tensor) [..., C]: Predictions.
        target (Tensor) [..., C]: Targets.

    Returns:
        Tensor [...]: The per-example sensitivity.
    """
    numer = torch.sum(input * target, -1)
    denom = torch.sum(target, -1)
    ratio = numer / denom
    return torch.where(denom.gt(0), ratio, torch.ones_like(ratio))


def specificity(input: Tensor, target: Tensor) -> Tensor:
    return sensitivity(1 - input, 1 - target)


def balanced_accuracy(input: Tensor, target: Tensor) -> Tensor:
    alpha = 0.5
    return (1-alpha) * sensitivity(input, target) + alpha * specificity(input, target)


def fdr(preds, y):
    return ((preds / y) == float('inf')).int()


def calc_gini(X, L, U, beta_min=0.0, beta_max=1.0):
    
    if np.sum(L == U) == len(L):
        mean_U = integrate_quantiles(X,U)
    else:
        mean_U = integrate_quantiles_upper(X, U)

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

