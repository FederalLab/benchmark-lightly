# @Author            : FederalLab
# @Date              : 2021-09-26 00:23:40
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:23:40
# Copyright (c) FederalLab. All rights reserved.

# type: ignore

import argparse
import os

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from openfed.data import FederatedDataset, samples_distribution

DEFAULT_TRAIN_FILE = 'fed_cifar100_train.h5'
DEFAULT_TEST_FILE = 'fed_cifar100_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMAGE = 'image'
_LABEL = 'label'


class CIFAR100(FederatedDataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform=None,
                 target_transform=None,
                 download: bool = False):
        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE if train else DEFAULT_TEST_FILE)
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f'{root} does not exist.')
        data_h5 = h5py.File(data_file, 'r')

        client_ids = list(data_h5[_EXAMPLE].keys())

        self.total_parts = len(client_ids)
        self.part_id = 0

        part_data_list = []
        part_target_list = []
        unique_label_set = set()
        for client_id in client_ids:
            # make core range is [0, 1]
            part_data_list.append(
                np.array(data_h5[_EXAMPLE][client_id][_IMAGE][()]) / 255.0)
            part_target_list.append(
                np.array(data_h5[_EXAMPLE][client_id][_LABEL][()]).squeeze())
            unique_label_set.update(part_target_list[-1].tolist())
        self.part_data_list = part_data_list
        self.part_target_list = part_target_list

        self.transform = transform
        self.target_transform = target_transform

        self.classes = len(unique_label_set)

    def __len__(self) -> int:
        return len(self.part_data_list[self.part_id])

    def __getitem__(self, index: int):
        data, target = self.part_data_list[
            self.part_id][index], self.part_target_list[self.part_id][index]

        data = torch.tensor(data).float().permute(2, 0, 1)
        target = torch.tensor(target).long()

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def total_samples(self):
        return sum([len(x) for x in self.part_data_list])


def get_cifar100(root, train: bool = True):
    root = os.path.join(root, 'raw')
    mean, std = [0.507, 0.486, 0.441], [0.267, 0.256, 0.276]
    if train:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(24),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return CIFAR100(root, train, transform=transform)