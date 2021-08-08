import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class Celeba(Model):
    def __init__(self, num_classes: int=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.logits = nn.Linear(32 * 5 * 5, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = top_one_acc

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return self.logits(conv4.view(conv4.size(0), -1))
