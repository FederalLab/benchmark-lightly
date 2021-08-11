import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class Mnist(Model):
    def __init__(self, num_classes: int=10):
        super().__init__()

        self.logits = nn.Linear(784, num_classes)

        self.loss_fn     = nn.CrossEntropyLoss()
        self.accuracy_fn = top_one_acc

    def forward(self, x):
        return self.logits(x.view(x.size(0), -1))
