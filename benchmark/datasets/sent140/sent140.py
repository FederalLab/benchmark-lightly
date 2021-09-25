# @Author            : FederalLab
# @Date              : 2021-09-26 00:24:51
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:24:51
# Copyright (c) FederalLab. All rights reserved.
import argparse
import os

from benchmark.datasets.simulation_dataset import SimulationDataset
from benchmark.datasets.utils.language_utils import (bag_of_words,
                                                     get_word_emb_arr,
                                                     line_to_indices,
                                                     val_to_vec)
from benchmark.datasets.utils.transforms import FloatTensor, LongTensor

os.sys.path.insert(0, '/Users/densechen/code/benchmark')

VOCAB_DIR = 'embs.json'


class Sent140(SimulationDataset):
    max_words: int = 25
    num_classes: int = 2

    def __init__(self, *args, task: str = 'bag_log_reg', **kwargs):
        assert task in ['bag_log_reg', 'stacked_lstm']
        super().__init__(*args, **kwargs)
        self.task = task
        _, self.indd, self.vocab = get_word_emb_arr(
            os.path.join(os.path.dirname(self.data_root), VOCAB_DIR))
        self.vocab_size = len(self.vocab)

    def __getitem__(self, index):
        data = self.data[self.parts[self.part_id]]

        x, y = data['x'][index], data['y'][index]  # type: ignore

        if self.task == 'bag_log_reg':
            x = bag_of_words(x[4], self.vocab)
            y = val_to_vec(self.num_classes, int(y))
        elif self.task == 'stacked_lstm':
            x = line_to_indices(x[4], self.indd, self.max_words)
            y = val_to_vec(self.num_classes, int(y))
        else:
            raise NotImplementedError('')

        if self.transform is not None:
            x = self.transform(x)
        if self.transform_target is not None:
            y = self.transform_target(y)
        return x, y


def get_sent140(root, task: str = 'bag_log_reg', train: bool = True):

    data_root = os.path.join(root, 'train' if train else 'test')

    return Sent140(data_root,
                   task=task,
                   transform=FloatTensor(),
                   transform_target=LongTensor())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_sent140(root=args.root, train=True)

    print(dataset)
    print(f'vocab size: {dataset.vocab_size}')

    # fetch data
    x, y = dataset[0]

    print(x.shape, y.shape)  # type: ignore
