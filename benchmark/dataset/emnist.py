from openfed.data import Analysis
from openfed.data.vision import EMNIST
from torchvision.transforms import ToTensor


def get_emnist(root, train: bool = True):
    return EMNIST(root, train=train, transform=ToTensor())


if __name__ == '__main__':
    dataset = get_emnist(root='data/Federated_EMNIST_TFF', train=True)
    Analysis.digest(dataset)
