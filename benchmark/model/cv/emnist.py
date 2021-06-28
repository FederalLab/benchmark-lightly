import torch
import torch.nn as nn
import torch.nn.functional as F


class EMNIST(nn.Module):
    """Used for classification.
    McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial Intelligence and Statistics. PMLR, 2017: 1273-1282.
    """

    def __init__(self, only_digits: bool = True):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(start_dim=1),
            nn.Linear(3136, 512, bias=True),
            nn.ReLU(),

            nn.Linear(512, 10 if only_digits else 62),
        )

    def forward(self, x):
        return self.classifier(x)


class EMNISTDropout(nn.Module):
    """Used for classification.
    Reddi S, Charles Z, Zaheer M, et al. Adaptive federated optimization[J]. arXiv preprint arXiv:2003.00295, 2020.
    """

    def __init__(self, only_digits: bool = True):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, bias=True),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, bias=True),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.25),

            nn.Flatten(start_dim=1),
            nn.Linear(9216, 128),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(128, 10 if only_digits else 62, bias=True),
        )

    def forward(self, x):
        return self.classifier(x)


class EMNISTLinear(nn.Module):
    """Used for classification.
    Reddi S, Charles Z, Zaheer M, et al. Adaptive federated optimization[J]. arXiv preprint arXiv:2003.00295, 2020.
    """

    def __init__(self, only_digits: bool = True):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(start_dim=1),

            nn.Linear(784, 200, bias=True),
            nn.ReLU(),

            nn.Linear(200, 200, bias=True),
            nn.ReLU(),

            nn.Linear(200, 10 if only_digits else 62, bias=True),
        )

    def forward(self, x):
        return self.linear(x)
