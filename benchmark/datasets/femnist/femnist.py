# @Author            : FederalLab
# @Date              : 2021-09-26 00:23:57
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:23:57
# Copyright (c) FederalLab. All rights reserved.
import argparse
import os

from benchmark.datasets.simulation_dataset import SimulationDataset
from benchmark.datasets.utils.transforms import FloatTensor, LongTensor

IMAGE_SIZE = 84


def get_femnist(root, train: bool = True):
    data_root = os.path.join(root, 'train' if train else 'test')

    return SimulationDataset(data_root, FloatTensor(), LongTensor())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_femnist(root=args.root, train=True)

    print(dataset)

    # fetch data
    x, y = dataset[0]

    print(x.shape, y.shape)
