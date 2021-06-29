from openfed.data import Analysis
from openfed.data.nlp import ShakespeareNCP, ShakespeareNWP
from torchvision.transforms import ToTensor


def get_shakespeare_ncp(root, train: bool = True):
    return ShakespeareNCP(root, train=train, transform=ToTensor())

def get_shakespeare_nwp(root, train: bool = True):
    return ShakespeareNWP(root, train=train, transform=ToTensor())


if __name__ == '__main__':
    dataset = get_shakespeare_ncp(root='data/Federated_Shakespeare_FedProx', train=True)
    Analysis.digest(dataset)

    dataset = get_shakespeare_nwp(root='data/Federated_Shakespeare_TFF', train=True)
    Analysis.digest(dataset)
