# @Author            : FederalLab
# @Date              : 2021-09-26 00:28:08
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:28:08
# Copyright (c) FederalLab. All rights reserved.

import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class Femnist(Model):
    """Used for classification.

    McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of
    deep networks from decentralized data[C]//Artificial Intelligence and
    Statistics. PMLR, 2017: 1273-1282.
    """
    def __init__(self, num_classes: int = 62):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten(start_dim=1)

        self.linear = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())

        self.logits = nn.Linear(512, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = top_one_acc

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        flatten = self.flatten(pool2)

        linear = self.linear(flatten)
        return self.logits(linear)
