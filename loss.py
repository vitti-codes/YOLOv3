import torch
import torch.nn as nn

from intersection_over_union import intersection_over_union

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        #Constants
        self.lamba_class = 1
        self.lamba_noobj = 10
        self.lamba_obg = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
