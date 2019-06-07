import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class WeightedBCELoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2

            loss = self.weights[1] * (target * torch.log(output)) + \
                   self.weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))


def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]


class FocalLossMulticlass(nn.Module):

    def __init__(self, alpha=1, gamma=2, eps=1e-7):
        super(FocalLossMulticlass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        num_cls = input.shape[1]
        model_out = torch.add(input, self.eps)
        onehot_labels = torch.nn.functional.one_hot(target, num_cls)
        ce = torch.mul(onehot_labels.float(), -torch.log(model_out))
        weight = torch.mul(onehot_labels.float(), torch.pow(torch.sub(1., model_out), self.gamma))
        fl = torch.mul(self.alpha, torch.mul(weight, ce))
        reduced_fl = torch.max(fl, dim=1)
        return reduced_fl
