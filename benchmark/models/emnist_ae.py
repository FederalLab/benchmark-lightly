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
