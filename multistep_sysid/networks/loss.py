import torch
import torch.nn.functional as F
import torch.nn as nn


class MSGELoss(nn.Module):
    def forward(self, predicted, true):
        """
        :param predicted: shape=(batch, time, _)
        :param true: shape=(batch, time, _)
        :return:
        """
        pred_grad = predicted[:, 1:, :] - predicted[:, :-1, :]
        true_grad = true[:, 1:, :] - predicted[:, :-1, :]
        mse = F.mse_loss(pred_grad, true_grad)
        return mse
