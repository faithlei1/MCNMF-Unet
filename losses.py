import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'Active_Contour_Loss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class Active_Contour_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        lenth term
        """

        x = input[:, :, 1:, :] - input[:, :, :-1, :]  # horizontal and vertical directions
        y = input[:, :, :, 1:] - input[:, :, :, :-1]

        delta_x = x[:, :, 1:, :-2] ** 2
        delta_y = y[:, :, :-2, 1:] ** 2
        delta_u = torch.abs(delta_x + delta_y)

        lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

        """
        region term
        """

        C_1 = torch.ones((96, 96)).cuda()
        C_2 = torch.zeros((96, 96)).cuda()

        region_in = torch.abs(torch.mean(input[:, 0, :, :] * ((target[:, 0, :, :] - C_1) ** 2)))  # equ.(12) in the paper
        region_in = region_in.cuda()
        region_out = torch.abs(torch.mean((1 - input[:, 0, :, :]) * ((target[:, 0, :, :] - C_2) ** 2)))  # equ.(12) in the paper
        region_out = region_out.cuda()

        lambdaP = 1  # lambda parameter could be various.
        mu = 1  # mu parameter could be various.

        return lenth + lambdaP * (mu * region_in + region_out)