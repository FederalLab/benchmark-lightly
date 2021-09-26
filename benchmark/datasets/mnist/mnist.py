# @Author            : FederalLab
# @Date              : 2021-09-26 00:24:12
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:24:12
# Copyright (c) FederalLab. All rights reserved.

import argparse
import os

from openfed.data import (IIDPartitioner, Partitioner, PartitionerDataset,
                          samples_distribution)
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def get_mnist(root,
              total_parts: int,
              train: bool = True,
              partitioner: Partitioner = None):
    root = os.path.join(root, 'raw')
    if not partitioner:
        partitioner = IIDPartitioner()
    return PartitionerDataset(MNIST(root, train, ToTensor(), download=True),
                              total_parts, partitioner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        type=str,
                        help='root directory',
                        default='data/raw')
    args = parser.parse_args()
    dataset = get_mnist(root=args.root, train=True, total_parts=100)
    samples_distribution(dataset)
