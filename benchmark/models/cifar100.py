# @Author            : FederalLab
# @Date              : 2021-09-26 00:28:03
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:28:03
# Copyright (c) FederalLab. All rights reserved.
"""mobilenetv2 in pytorch.

[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov,
    Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch.nn as nn
import torch.nn.functional as F

from .base import Model
from .utils import top_one_acc

# Modified from https://github.com/weiaicunzai/pytorch-cifar100/


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t,
                      in_channels * t,
                      3,
                      stride=stride,
                      padding=1,
                      groups=in_channels * t), nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True), nn.Conv2d(in_channels * t, out_channels,
                                              1), nn.BatchNorm2d(out_channels))

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class Cifar100(Model):
    def __init__(self, num_classes=100):
        super().__init__()

        self.pre = nn.Sequential(nn.Conv2d(3, 32, 1, padding=1),
                                 nn.BatchNorm2d(32), nn.ReLU6(inplace=True))

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(nn.Conv2d(320, 1280,
                                             1), nn.BatchNorm2d(1280),
                                   nn.ReLU6(inplace=True))

        self.conv2 = nn.Conv2d(1280, num_classes, 1)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = top_one_acc

    def forward(self, x):
        pre = self.pre(x)
        stage1 = self.stage1(pre)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        stage6 = self.stage6(stage5)
        stage7 = self.stage7(stage6)
        conv1 = self.conv1(stage7)
        pool = F.adaptive_avg_pool2d(conv1, 1)
        conv2 = self.conv2(pool)
        conv2 = conv2.view(conv2.size(0), -1)

        return conv2

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)
