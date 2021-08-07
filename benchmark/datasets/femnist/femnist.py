import argparse
import os
os.sys.path.insert(0, '/Users/densechen/code/benchmark')
from torchvision import transforms

from benchmark.datasets.simulation_dataset import SimulationDataset
from benchmark.datasets.utils.transforms import LoadImage, LongTensor

IMAGE_SIZE = 84


def get_femnit(root, train: bool = True):
    data_root = os.path.join(root, 'train' if train else 'test')

    return SimulationDataset(data_root, LongTensor(), LongTensor())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_femnit(root=args.root, train=True)

    print(dataset)

    # fetch data
    x, y = dataset[0]

    print(x.shape, y.shape)