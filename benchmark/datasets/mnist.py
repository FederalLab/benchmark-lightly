from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from openfed.data import (Analysis, IIDPartitioner, Partitioner,
                          PartitionerDataset)


def get_mnist(root, total_parts: int, train: bool = True, partitioner: Partitioner = None):
    if not partitioner:
        partitioner = IIDPartitioner()
    return PartitionerDataset(MNIST(root, train, ToTensor(), download=True), total_parts, partitioner)


if __name__ == '__main__':
    dataset = get_mnist(root='data', train=True, total_parts=100)
    Analysis.digest(dataset)
