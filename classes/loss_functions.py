import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight = None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth_size=1):

        """comment out if your model contains a sigmoid or equivalent activation layer"""
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        dice_loss = (2.*intersection + smooth_size)/(inputs.sum() + targets.sum() + smooth_size)

        return 1-dice_loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(DiceBCELoss, self).__init__()
    
    def forward(self, inputs, targets, smooth_size = 1):
        
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth_size) / (inputs.sum() + targets.sum() + smooth_size)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
