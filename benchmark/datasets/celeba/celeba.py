# @Author            : FederalLab
# @Date              : 2021-09-26 00:23:11
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:23:11
# Copyright (c) FederalLab. All rights reserved.
import argparse
import os

from torchvision import transforms

from ..simulation_dataset import SimulationDataset
from ..utils.transforms import BinaryTensor, LoadImage

IMAGE_SIZE = 84


def get_celeba(root, train: bool = True):
    transform = transforms.Compose([
        LoadImage(os.path.join(root, 'raw', 'img_align_celeba')),
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.ToTensor(),
    ])
    transform_target = BinaryTensor()

    data_root = os.path.join(root, 'train' if train else 'test')

    return SimulationDataset(data_root, transform, transform_target)
