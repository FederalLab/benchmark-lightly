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
from openfed.common.logging import logger
from openfed.data import Analysis, FederatedDataset
from openfed.data.utils import tar_xvf, wget_https

DEFAULT_TRAIN_FILE = 'fed_cifar100_train.h5'
DEFAULT_TEST_FILE = 'fed_cifar100_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMAGE = 'image'
_LABEL = 'label'


class CIFAR100(FederatedDataset):
    """Federated CIFAR100 Dataset from https://github.com/tensorflow/federated.

    Loads a federated version of the CIFAR-100 dataset.
    The dataset is downloaded and cached locally. If previously downloaded, it
    tries to load the dataset from cache.
    The dataset is derived from the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The training and testing examples are partitioned across 500 and 100 clients (respectively).
    No clients share any data samples, so it is a true partition of CIFAR-100.
    The train clients have string client IDs in the range [0-499], while the test
    clients have string client IDs in the range [0-99]. The train clients form a
    true partition of the CIFAR-100 training split, while the test clients form a
    true partition of the CIFAR-100 testing split.
    The data partitioning is done using a hierarchical Latent Dirichlet Allocation
    (LDA) process, referred to as the [Pachinko Allocation Method]
    (https://people.cs.umass.edu/~mccallum/papers/pam-icml06.pdf) (PAM).
    This method uses a two-stage LDA process, where each client has an associated
    multinomial distribution over the coarse labels of CIFAR-100, and a
    coarse-to-fine label multinomial distribution for that coarse label over the
    labels under that coarse label. The coarse label multinomial is drawn from a
    symmetric Dirichlet with parameter 0.1, and each coarse-to-fine multinomial
    distribution is drawn from a symmetric Dirichlet with parameter 10. Each
    client has 100 samples. To generate a sample for the client, we first select
    a coarse label by drawing from the coarse label multinomial distribution, and
    then draw a fine label using the coarse-to-fine multinomial distribution. We
    then randomly draw a sample from CIFAR-100 with that label (without
    replacement). If this exhausts the set of samples with this label, we
    remove the label from the coarse-to-fine multinomial and renormalize the
    multinomial distribution.
    Data set sizes:
        -   train: 50,000 examples
        -   test: 10,000 examples
    """
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform=None,
                 target_transform=None,
                 download: bool = False):
        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE if train else DEFAULT_TEST_FILE)
        if not os.path.isfile(data_file):
            if download:
                url = 'https://fedml.s3-us-west-1.amazonaws.com/fed_cifar100.tar.bz2'
                logger.debug(f'Download dataset from {url} to {root}')
                if wget_https(url, root):
                    if tar_xvf(os.path.join(root, 'fed_cifar100.tar.bz2'),
                               output_dir=root):
                        logger.debug('Downloaded.')
                else:
                    raise RuntimeError('Download dataset failed.')
            else:
                raise FileNotFoundError(f'{data_file} not exists.')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory')
    args = parser.parse_args()
    dataset = get_cifar100(root=args.root)
    Analysis.digest(dataset)
