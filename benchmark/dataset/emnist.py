from torchvision.transforms import ToTensor
from openfed.data.vision import EMNIST

def get_emnist(root, train: bool = True):
    return EMNIST(root, train=train, transform=ToTensor())


if __name__ == '__main__':
    dataset = get_emnist(root='data/Federated_EMNIST_TFF', train=True)

    for p in range(dataset.total_parts):
        dataset.set_part_id(p)
        print(f"Part: {p}, Total Samples: {len(dataset)}")
