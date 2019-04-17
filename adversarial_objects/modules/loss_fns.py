import torch
from torch import nn
import torch.nn.functional as F


def untargeted_loss_fn(pred_prob_y, true_y):
    return pred_prob_y[:, true_y].mean()


def targeted_loss_fn(pred_prob_y, true_y_topk, target_class, target_weight=-10.0):
    return (
        pred_prob_y[:, true_y_topk].mean(0).sum()
        + (target_weight * (pred_prob_y[:, target_class]).mean(0).sum())
    )
