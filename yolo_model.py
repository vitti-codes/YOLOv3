import torch
import torch.nn as nn


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
        self.leaky = nn.LeakyReLU()
        self.num_repeats = num_repeats
        self.in_channels = in_channels

    def forward(self, x):
        for i in range(self.num_repeats):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.leaky(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.leaky(x) #may not be necessary here

        self.in_channels = self.out_channels
        return x


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


class YOLO(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLO, self).__init__()

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

        self.scale1 = ScalePred(in_channels=1024, num_classes=num_classes)

        self.conv9 = MyConv(in_channels=out_channels, out_channels=256, kernel_size=1, stride=1)

        self.up1 = nn.Upsample(scale_factor=2)

        self.conv10 = MyConv(in_channels=26*26, out_channels=256, kernel_size=1, stride=1)
        self.conv11 = MyConv(in_channels=out_channels, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.scale2 = ScalePred(in_channels=512, num_classes=num_classes)

        self.conv12 = MyConv(in_channels=out_channels, out_channels=128, kernel_size=1, stride=1)

        self.up2 = nn.Upsample(scale_factor=2)

        self.conv13 = MyConv(in_channels=52*52, out_channels=128, kernel_size=1, stride=1)
        self.conv14 = MyConv(in_channels=out_channels, out_channels=256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        layers = []
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
        x = self.scale1(x)
        outputs.append(x)
        x = self.conv9(x)
        x = self.up1(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.scale2(x)
        outputs.append(x)
        x = self.conv12(x)
        x = self.up2(x)
        x = self.conv13(x)
        x = self.conv14(x)

        return x #or return outputs?




if __name__ == "__main__":
    num_classes = 80
    IMAGE_SIZE = 416
    model = YOLO(in_channels=416*416, out_channels=416*416)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)) #import image
    out = model(x)

