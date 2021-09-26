# @Author            : FederalLab
# @Date              : 2021-09-26 00:23:57
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:23:57
# Copyright (c) FederalLab. All rights reserved.
import os

from ..simulation_dataset import SimulationDataset
from ..utils.transforms import FloatTensor, LongTensor

IMAGE_SIZE = 84


def get_femnist(root, train: bool = True):
    data_root = os.path.join(root, 'train' if train else 'test')

    return SimulationDataset(data_root, FloatTensor(), LongTensor())
