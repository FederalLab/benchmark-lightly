# @Author            : FederalLab
# @Date              : 2021-09-26 00:25:54
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:25:54
# Copyright (c) FederalLab. All rights reserved.
import openfed.data as data

from .utils.util import read_dir


class SimulationDataset(data.FederatedDataset):
    """parses data in given train and test data directories.

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    def __init__(self, data_root: str, transform=None, transform_target=None):
        super().__init__()
        self.data_root = data_root

        self.parts, self.groups, self.data = read_dir(data_root)

        self.total_parts = len(self.parts)

        self.transform = transform
        self.transform_target = transform_target

    @property
    def total_samples(self):
        return sum([len(x['x']) for x in self.data.values()])  # type:ignore

    def __len__(self):
        return len(self.data[self.parts[self.part_id]]['x'])  # type:ignore

    def __getitem__(self, index):
        data = self.data[self.parts[self.part_id]]

        x, y = data['x'][index], data['y'][index]  # type: ignore

        if self.transform is not None:
            x = self.transform(x)
        if self.transform_target is not None:
            y = self.transform_target(y)

        return x, y
