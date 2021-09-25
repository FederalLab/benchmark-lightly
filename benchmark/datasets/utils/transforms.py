# @Author            : FederalLab
# @Date              : 2021-09-26 00:34:21
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:34:21
# Copyright (c) FederalLab. All rights reserved.
import os

import torch
from PIL import Image


class LoadImage(object):
    def __init__(self, basedir=''):
        self.basedir = basedir

    def __call__(self, image_path: str):
        return Image.open(os.path.join(self.basedir,
                                       image_path)).convert('RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LongTensor(object):
    def __call__(self, x):
        return torch.tensor(x).long()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FloatTensor(object):
    def __call__(self, x):
        return torch.tensor(x).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class BinaryTensor(object):
    def __call__(self, x):
        return torch.tensor(x > 0).long()

    def __repr__(self):
        return self.__class__.__name__ + '()'
