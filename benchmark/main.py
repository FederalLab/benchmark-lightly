import os
import sys
sys.path.insert(0, "/Users/densechen/code/OpenFed")

import openfed
from openfed.data.vision import EMNIST
from torchvision.transforms import ToTensor

dataset = EMNIST('.', train=True, transform=ToTensor())


for p in range(dataset.total_parts):
    dataset.set_part_id(p)
    print(f"Part: {p}, Total Samples: {len(dataset)}")