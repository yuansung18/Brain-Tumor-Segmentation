import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models.utils import get_tensor_from_array
from .dice import dice_score_hub
from .utils import GetClassWeights
from utils import to_one_hot_label


def ce_minus_log_dice(logits: torch.Tensor, tar: np.array, dice_type: str = 'my'):
    crossentropy_loss, log_1 = weighted_cross_entropy(logits, tar)

    dice_fn = dice_score_hub[dice_type]
    onehot_tar = to_one_hot_label(tar, class_num=logits.shape[1])
    dice_score, log_2 = dice_fn(logits, onehot_tar)

    total_loss = crossentropy_loss - torch.log(dice_score)
    return total_loss, {**log_1, **log_2}


def weighted_cross_entropy(logits: torch.Tensor, target: np.array):
    weights = GetClassWeights()(target, class_num=logits.shape[1])
    weights = get_tensor_from_array(weights)
    target = get_tensor_from_array(target).long()
    loss = nn.CrossEntropyLoss(weight=weights)(logits, target)
    return loss, {'crossentropy_loss': loss.item()}


def minus_dice(logits: torch.Tensor, tar: np.array, dice_type: str = 'my'):
    if not dice_type == 'sigmoid':
        onehot_tar = to_one_hot_label(tar, class_num=logits.shape[1])
    else:
        onehot_tar = tar
    dice_fn = dice_score_hub[dice_type]
    dice_score, log = dice_fn(logits, onehot_tar)
    return -dice_score, log


def l2_plus_kl(logits: torch.Tensor, target: np.array, mean, var, weight_L2: float = 0.1, weight_KL: float = 0.1):
    total_node = logits.view(logits[0], -1).shape[-1]
    target = get_tensor_from_array(target)

    loss_L2 = torch.mean((logits - target) ** 2)
    loss_KL = torch.sum(torch.exp(var) + mean ** 2 - 1. - var) / total_node
    loss = weight_L2 * loss_L2 + weight_KL * loss_KL
    return loss, {'L2_KL_loss': loss.item()}
