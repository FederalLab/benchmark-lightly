# @Author            : FederalLab
# @Date              : 2021-09-26 00:27:51
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:27:51
# Copyright (c) FederalLab. All rights reserved.
from .base import Model
from .celeba import Celeba
from .cifar100 import Cifar100
from .femnist import Femnist
from .mnist import Mnist
from .reddit import Reddit
from .sent140 import Sent140
from .shakespeare import Shakespeare
from .stackoverflow import Stackoverflow
from .synthetic import Synthetic


def build_model(name, *args, **kwargs):
    if name == 'celeba':
        return Celeba(*args, **kwargs)
    elif name == 'cifar100':
        return Cifar100(*args, **kwargs)
    elif name == 'femnist':
        return Femnist(*args, **kwargs)
    elif name == 'mnist':
        return Mnist(*args, **kwargs)
    elif name == 'reddit':
        return Reddit(*args, **kwargs)
    elif name == 'sent140':
        return Sent140(*args, **kwargs)
    elif name == 'shakespeare':
        return Shakespeare(*args, **kwargs)
    elif name == 'stackoverflow':
        return Stackoverflow(*args, **kwargs)
    elif name == 'synthetic':
        return Synthetic(*args, **kwargs)
    else:
        raise ValueError('Unknown model.')


__all__ = [
    'Model',
    'build_model',
]
