# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
