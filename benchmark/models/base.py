# @Author            : FederalLab
# @Date              : 2021-09-26 00:27:54
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:27:54
# Copyright (c) FederalLab. All rights reserved.

from typing import Callable

import torch.nn as nn


class Model(nn.Module):
    loss_fn: Callable
    accuracy_fn: Callable
