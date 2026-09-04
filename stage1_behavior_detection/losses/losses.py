# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
import torch
import torch.nn as nn
import torch.nn.functional as F

class_weights = torch.FloatTensor([0.16, 0.84])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)  # Calculate the mean before returning
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        elif self.reduction == 'weighted':
            weight_matrix = torch.where(targets == 0, 0.16, 0.84)
            weight_matrix = weight_matrix / torch.sum(weight_matrix)

            return (torch.sum(torch.mul(focal_loss, weight_matrix)))
        else:
            return focal_loss