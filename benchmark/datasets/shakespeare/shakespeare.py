# @Author            : FederalLab
# @Date              : 2021-09-26 10:19:57
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 10:19:57
# Copyright (c) FederalLab. All rights reserved.

import os

from ..simulation_dataset import SimulationDataset
from ..utils.language_utils import NUM_LETTERS, letter_to_vec, word_to_indices
from ..utils.transforms import FloatTensor, LongTensor

VOCAB_DIR = 'embs.json'


class Shakespeare(SimulationDataset):
    vocab_size: int = NUM_LETTERS

    def __getitem__(self, index):
        data = self.data[self.parts[self.part_id]]

        x, y = data['x'][index], data['y'][index]  # type: ignore

        x = word_to_indices(x)
        y = letter_to_vec(y)

        if self.transform is not None:
            x = self.transform(x)
        if self.transform_target is not None:
            y = self.transform_target(y)
        return x, y


def get_shakespeare(root, train: bool = True):

    data_root = os.path.join(root, 'train' if train else 'test')

    return Shakespeare(data_root, FloatTensor(), LongTensor())
