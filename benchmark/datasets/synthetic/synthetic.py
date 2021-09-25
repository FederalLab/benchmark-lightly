# @Author            : FederalLab
# @Date              : 2021-09-26 00:25:44
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:25:44
# Copyright (c) FederalLab. All rights reserved.
import argparse
import os

from torchvision import transforms

from benchmark.datasets.simulation_dataset import SimulationDataset
from benchmark.datasets.utils.transforms import FloatTensor, LongTensor

IMAGE_SIZE = 84


def get_synthetic(root, train: bool = True):
    data_root = os.path.join(root, 'train' if train else 'test')
    return SimulationDataset(data_root, FloatTensor(), LongTensor())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_synthetic(root=args.root, train=True)

    print(dataset)

    # fetch data
    x, y = dataset[0]

    print(x.shape, y.shape)
