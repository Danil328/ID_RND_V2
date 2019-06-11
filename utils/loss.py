import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


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
            # weights = (targets + 1) / 2.0
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


class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        weight = Variable(self.weight)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.0)

        elif type == 'softmax':
            B, C = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.0)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        focus = torch.clamp(focus, 0, 2)

        batch_loss = -class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss