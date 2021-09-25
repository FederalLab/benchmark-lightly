# @Author            : FederalLab
# @Date              : 2021-09-26 00:24:36
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:24:36
# Copyright (c) FederalLab. All rights reserved.
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np

from benchmark.datasets.simulation_dataset import SimulationDataset
from benchmark.datasets.utils.transforms import FloatTensor, LongTensor

VOCABULARY_PATH = 'reddit_vocab.pck'


class Reddit(SimulationDataset):
    """You should put the `reddit_vocab.pck` under
    `data_root/reddit_vocab.pck`, but not `data_root/xxx/reddit_vocab.pck`.

    It will take a quite long time to load data from disk.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = self.load_vocab(
        )

    def load_vocab(self):
        vocab_file = pickle.load(
            open(
                os.path.join(os.path.dirname(self.data_root), VOCABULARY_PATH),
                'rb'))
        vocab = defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])

        return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file[
            'pad_symbol']

    def _tokens_to_ids(self, tokens):
        return np.array([self.vocab[word] for word in tokens])

    def __getitem__(self, index):
        data = self.data[self.parts[self.part_id]]
        x, y = data['x'][index], data['y'][index]  # type: ignore

        x, y = self._tokens_to_ids(x[0]), self._tokens_to_ids(
            y['target_tokens'][0])

        if self.transform is not None:
            x = self.transform(x)
        if self.transform_target is not None:
            y = self.transform_target(y)
        return x, y


def get_reddit(root, mode: str = 'train'):
    assert mode in ['train', 'val', 'test']
    data_root = os.path.join(root, mode)

    return Reddit(data_root, FloatTensor(), LongTensor())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_reddit(root=args.root, mode='val')

    print(dataset)
    print(f'vocab size: {dataset.vocab_size}')

    # fetch data
    x, y = dataset[0]

    print(x.shape, y.shape)
