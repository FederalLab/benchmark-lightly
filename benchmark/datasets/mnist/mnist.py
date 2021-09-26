# @Author            : FederalLab
# @Date              : 2021-09-26 00:24:12
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:24:12
# Copyright (c) FederalLab. All rights reserved.
import os

from openfed.data import IIDPartitioner, Partitioner, PartitionerDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from ..utils.transforms import LongTensor


def get_mnist(root,
              total_parts: int,
              train: bool = True,
              partitioner: Partitioner = None):
    root = os.path.join(root, 'raw')
    if not partitioner:
        partitioner = IIDPartitioner()
    return PartitionerDataset(
        MNIST(root,
              train,
              transform=ToTensor(),
              target_transform=LongTensor(),
              download=True), total_parts, partitioner)
