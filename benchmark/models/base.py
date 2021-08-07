from abc import abstractmethod
from typing import Callable

import torch.nn as nn


class Model(nn.Module):
    loss_fn: Callable
    acc_fn: Callable
