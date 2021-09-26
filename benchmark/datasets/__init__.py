# @Author            : FederalLab
# @Date              : 2021-09-26 00:25:50
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:25:50
# Copyright (c) FederalLab. All rights reserved.

from .simulation_dataset import SimulationDataset

del simulation_dataset


def build_dataset(name, *args, **kwargs):
    if name == 'celeba':
        from .celeba import get_celeba
        return get_celeba(*args, **kwargs)
    elif name == 'cifar100':
        from .cifar100 import get_cifar100
        return get_cifar100(*args, **kwargs)
    elif name == 'femnist':
        from .femnist import get_femnist
        return get_femnist(*args, **kwargs)
    elif name == 'mnist':
        from .mnist import get_mnist
        return get_mnist(*args, **kwargs)
    elif name == 'reddit':
        from .reddit import get_reddit
        return get_reddit(*args, **kwargs)
    elif name == 'sent140':
        from .sent140 import get_sent140
        return get_sent140(*args, **kwargs)
    elif name == 'shakespeare':
        from .shakespeare import get_shakespeare
        return get_shakespeare(*args, **kwargs)
    elif name == 'stackoverflow':
        from .stackoverflow import get_stackoverflow
        return get_stackoverflow(*args, **kwargs)
    elif name == 'synthetic':
        from .synthetic import get_synthetic
        return get_synthetic(*args, **kwargs)
    else:
        raise ValueError('Unknown dataset')
