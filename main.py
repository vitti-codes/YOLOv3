
import torch
import torchvision
from torchvision import transforms
#from dataset import mydataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO



#input image is (416, 416, 3)

#output of network will be  [(52, 52, 3, (4 + 1 + num_classes)),
# (26, 26, 3, (4 + 1 + num_classes)), (13, 13, 3, (4 + 1 + num_classes))]

#Hyper Parameters
image_size = 1
batch_size = 10

#Loading the dataset

dataDir='C:/Users/vitti/Desktop/mscoco_set/train2017'
dataType='train_2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)

train = torchvision.datasets.CocoDetection(root=dataDir, annFile=annFile)
train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True,)


dataDir_val='C:/Users/vitti/Desktop/mscoco_set/val2017'
dataType_val='val_2017'
annFile_val='{}/annotations/instances_{}.json'.format(dataDir, dataType_val)

val = torchvision.datasets.CocoDetection(root=dataDir_val, annFile=annFile_val)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)



transform = transforms.Compose([transforms.Resize(300),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#training