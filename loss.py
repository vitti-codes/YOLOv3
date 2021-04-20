import torch
import torch.nn as nn

from intersection_over_union import intersection_over_union


'''
The original paper of Redmon et al does not give a clear explanation of the loss, however it can be deducted
that this loss function is subdivided into 3 major components. Namely, the Regression loss, the confidence loss and the
classification loss. 

Confidence Loss: which determines whether there are objects in the prediction frame
Regression loss: calculated when box contains objects
Classification loss: determines which category the object in the prediction frame belongs to 




CONFIDENCE LOSS--> get max value of iou, if the largest iou is less than threshold, it is considered that  the 
prediction has no objects, which means it is a background box. 
calc conf loss: see loss func in paper

CLASSIFICATION LOSS --> cross entropy of the true classification 

Reg loss -->if the grid cell contains an object, then the bounding box lss will be calculated 


'''
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
        Loss = Lam