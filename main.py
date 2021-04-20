
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
transform = transforms.Compose([transforms.Resize(416*416),
                                transforms.ToTensor()])

root_train = '/Users/paolocastelnuovo/Downloads/train2017'
ann_train= '/Users/paolocastelnuovo/Downloads/annotations/instances_train2017.json'
train = torchvision.datasets.CocoDetection(root=root_train,  annFile=ann_train, transforms=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

root_val = '/Users/paolocastelnuovo/Downloads/val2017'
ann_val= '/Users/paolocastelnuovo/Downloads/annotations/instances_val2017.json'
val = torchvision.datasets.CocoDetection(root=root_val,  annFile=ann_val, transforms=transform)
val_loader = torch.utils.data.DataLoader(val, batch_size=10, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#training


if __name__ == "__main__":
    num_classes = 80
    IMAGE_SIZE = 416
    batch_size = 1


    import yolo_model.
    import tqdm as tqdm
    import torch.optim as optim
    from pytorchyolo.utils.loss import compute_loss
    from pytorchyolo.models import load_model
    from pytorchyolo.models import Darknet

    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    learning_rate = 1e-6
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(lr=model.hyperparams['learning rate'], params=params)

    epochs = 10

    for epoch in range(epochs):
        print('training model....')
        model = load_model(model_path='yolo_model.py')
        model.train()
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch{epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        print(epoch+1)
        




