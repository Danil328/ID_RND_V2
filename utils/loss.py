import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, add_weight=False, pos_weight=1.0, neg_weight=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.add_weight = add_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, inputs, targets):
        if self.add_weight:
            weights = targets.clone()
            weights[(targets == 0.0) & (inputs >= 0.5)] = self.pos_weight
            weights[(targets == 0.0) & (inputs < 0.5)] = self.pos_weight
            weights[(targets == 1.0) & (inputs >= 0.5)] = self.pos_weight
            weights[(targets == 1.0) & (inputs < 0.5)] = self.neg_weight
        else:
            weights = None

        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, weight=weights)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False, weight=weights)

        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss
