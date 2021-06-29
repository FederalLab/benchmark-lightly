from torchvision.transforms import ToTensor
from openfed.data.nlp import StackOverFlowNWP, StackOverFlowTP
from openfed.data import Analysis

def get_stackoverflow_nwp(root, train: bool = True):
    return StackOverFlowNWP(root, train=train, transform=ToTensor())


def get_stackoverflow_tp(root, train: bool = True):
    return StackOverFlowTP(root, train=train, transform=ToTensor())


if __name__ == '__main__':
    dataset = get_stackoverflow_nwp(
        root='data/Federated_StackOverFlow_TFF', train=True)
    Analysis.digest(dataset)

    dataset = get_stackoverflow_tp(
        root='data/Federated_StackOverFlow_TFF', train=True)
    Analysis.digest(dataset)
