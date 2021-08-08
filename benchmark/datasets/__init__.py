from .celeba import get_celeba
from .cifar100 import get_cifar100
from .femnist import get_femnist
from .mnist import get_mnist
from .reddit import get_reddit
from .sent140 import get_sent140
from .shakespeare import get_shakespeare
from .simulation_dataset import SimulationDataset
from .stackoverflow import get_stackoverflow
from .synthetic import get_synthetic

del simulation_dataset
del celeba
del cifar100
del femnist
del mnist
del reddit
del sent140
del shakespeare
del stackoverflow
del synthetic


def build_dataset(name, *args, **kwargs):
    if name == 'celeba':
        return get_celeba(*args, **kwargs)
    elif name == 'cifar100':
        return get_cifar100(*args, **kwargs)
    elif name == 'femnist':
        return get_femnist(*args, **kwargs)
    elif name == 'mnist':
        return get_mnist(*args, **kwargs)
    elif name == 'reddit':
        return get_reddit(*args, **kwargs)
    elif name == 'sent140':
        return get_sent140(*args, **kwargs)
    elif name == 'shakespeare':
        return get_shakespeare(*args, **kwargs)
    elif name == 'stackoverflow':
        return get_stackoverflow(*args, **kwargs)
    elif name == 'synthetic':
        return get_synthetic(*args, **kwargs)
    else:
        raise ValueError('Unknown dataset')
