import argparse
import os
os.sys.path.insert(0, '/Users/densechen/code/benchmark')
from torchvision import transforms

from benchmark.datasets.simulation_dataset import SimulationDataset
from benchmark.datasets.utils.transforms import LoadImage, LongTensor

IMAGE_SIZE = 84


def get_celeba(root, train: bool = True):
    transform = transforms.Compose(
        [
            LoadImage(os.path.join(root, 'raw', 'img_align_celeba')),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    transform_target = LongTensor()

    data_root = os.path.join(root, 'train' if train else 'test')

    return SimulationDataset(data_root, transform, transform_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_celeba(root=args.root, train=True)

    print(dataset)

    # fetch data
    x, y = dataset[0]

    print(x.shape, y.shape)