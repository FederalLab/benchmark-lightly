import os

from PIL import Image
import torch


class LoadImage(object):
    def __init__(self, basedir=''):
        self.basedir = basedir

    def __call__(self, image_path: str):
        return Image.open(os.path.join(self.basedir, image_path)).convert("RGB")

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LongTensor(object):
    def __call__(self, x):
        return torch.tensor(x).long()

    def __repr__(self):
        return self.__class__.__name__ + '()'