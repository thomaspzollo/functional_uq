import random
import numpy as np
from torch import Tensor
import torch
from dataclasses import asdict, dataclass
from collections import OrderedDict
import math
from tqdm import tqdm


def calibrator(probs, thresholds, eps=1e-6):
    
    probs = probs.unsqueeze(1)
    x = np.array(probs, dtype=np.float64)
    x = np.clip(x, eps, 1 - eps)
    x = np.log(x / (1 - x))
    x = x * np.array(thresholds)
    output = 1 / (1 + np.exp(-x))
    return output


@dataclass
class Split:
    X: torch.Tensor
    g: torch.Tensor


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def predict(thresholds: Tensor, scores: Tensor) -> Tensor:
    """Use the provided thresholds to create prediction sets from the given
    scores.
    Args:
        thresholds (Tensor) [H] (float): Input thresholds.
        scores (Tensor) [...] (float): Scores to be thresholded.
    Returns:
        Tensor [H, ...] (long): Binary mask of predictions, constructed by thresholding
        the scores.
    """
    preds = torch.gt(scores.unsqueeze(-1), thresholds)
    return torch.permute(preds, [-1] + list(range(scores.dim()))).long()
