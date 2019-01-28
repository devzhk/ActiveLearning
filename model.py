import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            stride = 2
        else:
            stride = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=1),
        )
        self.relu = nn.ReLU(inplace=True)
        if self.downsample:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=1),
            )

    def forward(self, x):
        out = self.conv1(x)
        if self.downsample:
            x = self.conv2(x)
        out = x + out
        return self.relu(out)


class Model(nn.Module):
    def __init__(self, in_channels, class_num=10, show=False):
        super().__init__()
        self.show = show
        self.block1 = nn.Conv2d(in_channels, 64, 7, 2)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.block3 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128),
        )
        self.block4 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256)
        )
        self.block5 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512),
            nn.AvgPool2d(3)
        )
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.block1(x)
        if self.show:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.show:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.show:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.show:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.show:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
