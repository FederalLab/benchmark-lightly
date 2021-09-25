# @Author            : FederalLab
# @Date              : 2021-09-26 00:28:47
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:28:47
# Copyright (c) FederalLab. All rights reserved.
import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class Synthetic(Model):
    def __init__(self, num_classes: int = 5, input_dim: int = 60):
        super().__init__()

        self.logits = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy_fn = top_one_acc

    def forward(self, x):
        return self.logits(x)
