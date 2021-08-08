from .celeba import get_celeba
from .cifar100 import get_cifar100
from .femnist import get_femnit
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
