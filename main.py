
import torch
import torchvision
from torchvision import transforms

import os
import os.path

from torch.utils.data import DataLoader
import COCO_loader
from pycocotools.coco import COCO

import yolo_model
#import loss
#import train
#import mean_avg_precision
#import intersection_over_union

#torchvision.models.resnet101(pretrained=True)

#input image is (416, 416, 3)

#output of network will be  [(52, 52, 3, (4 + 1 + num_classes)),
# (26, 26, 3, (4 + 1 + num_classes)), (13, 13, 3, (4 + 1 + num_classes))]




#Loading the data

root_train = '/Users/paolocastelnuovo/Downloads/train2017'
ann_train= '/Users/paolocastelnuovo/Downloads/annotations/instances_train2017.json'
train = torchvision.datasets.CocoDetection(root=root_train,  annFile=ann_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

root_val = '/Users/paolocastelnuovo/Downloads/val2017'
ann_val= '/Users/paolocastelnuovo/Downloads/annotations/instances_val2017.json'
val = torchvision.datasets.CocoDetection(root=root_val,  annFile=ann_val)
val_loader = torch.utils.data.DataLoader(val, batch_size=10, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#training


if __name__ == "__main__":
    num_classes = 80
    IMAGE_SIZE = 416
    batch_size = 1
    model = Net(train)
    #x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)) #import image
    #out = model(x)

