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


from typing import List

import torch
import torch.nn as nn


class EMNISTAE(nn.Module):
    """Recommended model to use for EMNIST AutoEncoder experiments.

    Reddi S, Zaheer M, Sachan D, et al. Adaptive methods for nonconvex optimization[C]//Proceeding of 32nd Conference on Neural Information Processing Systems (NIPS 2018). 2018.
    """

    def __init__(self, out_channel: List[int] = None):
        super().__init__()
        if not out_channel:
            out_channel = [784, 1000, 500, 250, 30]

        prev_channels = out_channel[0]
        layer_list = []
        for out_ch in out_channel[1:-1]:
            layer_list.append(nn.Linear(prev_channels, out_ch))
            layer_list.append(nn.Sigmoid())

            prev_channels = out_ch
        layer_list.append(nn.Linear(prev_channels, out_channel[-1]))

        out_channel.reverse()
        prev_channels = out_channel[0]
        for out_ch in out_channel[1:]:
            layer_list.append(nn.Linear(prev_channels, out_ch))
            layer_list.append(nn.Sigmoid())
            prev_channels = out_ch
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(torch.flatten(x, start_dim=1)).reshape(*x.shape)
