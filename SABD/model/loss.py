import logging

from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeculoiuLoss(_Loss):
    """
    Learning Text Similarity with Siamese Recurrent Networks - Paul Neculoiu, Maarten Versteegh and Mihai Rotaru

    Contrastive loss for cosine similarity
    """

    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        super(NeculoiuLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, similarity, y):
        lossPos = y * 0.25 * (1 - similarity) ** 2
        lossNegative = (1 - y) * torch.where(similarity < self.margin, torch.zeros(similarity.shape), similarity) ** 2
        loss = lossPos + lossNegative

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class CosineLoss(_Loss):

    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        super(CosineLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, similarity, y):
        loss = y * (1 - similarity) + (1 - y) * F.relu(similarity - self.margin)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class TripletLoss(_Loss):

    def __init__(self, margin=1, size_average=None, reduce=None, reduction='mean'):
        super(TripletLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.logger = logging.getLogger(__name__)

    def forward(self, output, target):
        simAnchorPos, simAnchorNeg = output

        rs = self.margin - simAnchorPos + simAnchorNeg
        loss = F.relu(rs)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
