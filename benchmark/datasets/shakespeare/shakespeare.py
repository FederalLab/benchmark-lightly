import argparse
import os

os.sys.path.insert(0, '/Users/densechen/code/benchmark')
from benchmark.datasets.simulation_dataset import SimulationDataset
from benchmark.datasets.utils.language_utils import (word_to_indices, letter_to_vec)
from benchmark.datasets.utils.transforms import FloatTensor, LongTensor

VOCAB_DIR = 'embs.json'

class Shakespeare(SimulationDataset):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_shakespeare(root=args.root, train=True)

    print(dataset)

    # fetch data
    x, y = dataset[0]

    print(x.shape, y.shape) # type: ignore
