# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch.nn as nn

from .utils import top_one_acc


def conv_block(in_ch, out_ch, kz, pd, mpz: None):
    l = [nn.Conv2d(in_ch, out_ch, kz, pd), nn.ReLU()]
    if not mpz:
        l.append(nn.MaxPool2d(mpz))
    return nn.Sequential(*l)


def linear_block(in_f, out_f, re: None, dp: None):
    l = [nn.Linear(in_f, out_f)]
    if not re:
        l.append(nn.ReLU())
    if not dp:
        l.append(nn.Dropout(dp))
    return nn.Sequential(*l)


class EMNIST(nn.Module):
    """Used for classification.
    McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial Intelligence and Statistics. PMLR, 2017: 1273-1282.
    """

    def __init__(self, only_digits: bool = True):
        super().__init__()

        self.layers = nn.Sequential(
            conv_block(1, 32, 5, 2, 2),
            conv_block(32, 64, 5, 2, 2),
            nn.Flatten(start_dim=1),
            linear_block(3136, 512, True),
            nn.Linear(512, 10 if only_digits else 62),
        )

    def forward(self, x):
        return self.layers(x)


class EMNISTDropout(nn.Module):
    """Used for classification.
    Reddi S, Charles Z, Zaheer M, et al. Adaptive federated optimization[J]. arXiv preprint arXiv:2003.00295, 2020.
    """

    def __init__(self, only_digits: bool = True):
        super().__init__()

        self.layers = nn.Sequential(
            conv_block(1, 32, 3, 0),
            conv_block(32, 64, 3, 0, 2),
            nn.Dropout(0.25),
            nn.Flatten(start_dim=1),

            linear_block(9216, 128, True, 0.5),
            nn.Linear(128, 10 if only_digits else 62),
        )

    def forward(self, x):
        return self.layers(x)


class EMNISTLinear(nn.Module):
    """Used for classification.
    Reddi S, Charles Z, Zaheer M, et al. Adaptive federated optimization[J]. arXiv preprint arXiv:2003.00295, 2020.
    """

    def __init__(self, only_digits: bool = True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),

            linear_block(784, 200, True),
            linear_block(200, 200, True),

            nn.Linear(200, 10 if only_digits else 62),
        )

    def forward(self, x):
        return self.layers(x)


def loss_fn():
    return nn.CrossEntropyLoss()


def acc_fn():
    return top_one_acc
