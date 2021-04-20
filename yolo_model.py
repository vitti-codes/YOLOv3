import torch
import torch.nn as nn
import pytorchyolo


class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,):
        super(MyConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.batch_norm(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(out_channels)
        self.num_repeats = num_repeats
        self.in_channels = in_channels

    def forward(self, x):
        x_prev = x
        
        for i in range(self.num_repeats):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.linear(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.linear(x)
        self.in_channels = self.out_channels

        return nn.concat(x_prev, x)
        


class ScalePred(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePred, self).__init__()
        self.num_classes = num_classes
        self.pred = nn.Conv2d(in_channels, out_channels=85*3, kernel_size=1)

    def forward(self, x):
        y = self.pred(x)
        y = y.reshape([0], 3, self.num_classes+5, x.shape[2], x.shape[3])
        y = y.permute(0, 1, 3, 4, 2)

        return y

class Detect(nn.Module):
    '''yolo layers for the network, takes the anchors
    each cell makes 3 detections, thus each cell needs 3 anchor boxes'''
    def __init__(self, in_channels, out_channels, mask):
        super(Detect, self).__init__()
    #must be 1 of these 3 masks (specified as parameter
        self.mask1 = 6, 7, 8
        self.mask2 = 3, 4, 5
        self.mask3 = 0, 1, 2

        self.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]

    # returns: [(10, 13), (16, 30), (33, 23)]

    def forward(self, x, anchors):
        if self.mask1 == self.mask:
            self.anchors = [anchors[i] for i in self.mask]
        elif self.mask2 == self.mask:
            self.anchors = [anchors[i] for i in self.mask]
        elif self.mask3 == self.mask:
            self.anchors = [anchors[i] for i in self.mask]
        return self.anchors
    #detection layer which takes in parameter of anchors

class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()

        self.conv1 = MyConv(in_channels=in_channels, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.conv2 = MyConv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.res1 = ResBlock(in_channels=out_channels, out_channels=out_channels, num_repeats=1)
        self.conv3 = MyConv(in_channels=out_channels, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.res2 = ResBlock(in_channels=out_channels, out_channels=256, num_repeats=2)

        self.conv4 = MyConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

        self.res3 = ResBlock(in_channels=out_channels, out_channels=out_channels, num_repeats=8)
        self.conv5 = MyConv(in_channels=out_channels, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.res4 = ResBlock(in_channels=out_channels, out_channels=out_channels, num_repeats=8)
        self.conv6 = MyConv(in_channels=out_channels, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.res5 = ResBlock(in_channels=out_channels, out_channels=out_channels, num_repeats=4)
        #end of darknet

        self.conv7 = MyConv(in_channels=out_channels, out_channels=512, kernel_size=1, stride=1)
        self.conv8 = MyConv(in_channels=out_channels, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.detect1 = Detect(in_channels=out_channels, out_channels=out_channels, mask=mask1)

        self.scale1 = ScalePred(in_channels=1024, num_classes=num_classes)
        self.conv9 = MyConv(in_channels=out_channels, out_channels=256, kernel_size=1, stride=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv10 = MyConv(in_channels=26*26, out_channels=256, kernel_size=1, stride=1)
        self.conv11 = MyConv(in_channels=out_channels, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.detect2 = Detect(in_channels=out_channels, out_channels=out_channels, mask=mask2)

        self.scale2 = ScalePred(in_channels=512, num_classes=num_classes)
        self.conv12 = MyConv(in_channels=out_channels, out_channels=128, kernel_size=1, stride=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv13 = MyConv(in_channels=52*52, out_channels=128, kernel_size=1, stride=1)
        self.conv14 = MyConv(in_channels=out_channels, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.detect3 = Detect(in_channels=out_channels, out_channels=out_channels, mask=mask3)

    def forward(self, x):
        #layers = []
        outputs = []


        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        x = self.conv5(x)
        x = self.res4(x)
        x = self.conv6(x)
        x = self.res5(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.detect1(x)
        x = self.scale1(x)
        outputs.append(x)
        x = self.conv9(x)
        x = self.up1(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.detect2(x)
        x = self.scale2(x)
        outputs.append(x)
        x = self.conv12(x)
        x = self.up2(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.detect3(x)
        
        return x 




