from torch import Tensor
import torch
from dataclasses import asdict, dataclass
from collections import OrderedDict
import math
from tqdm import tqdm


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
