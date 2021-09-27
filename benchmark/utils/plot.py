# @Author            : FederalLab
# @Date              : 2021-09-26 00:29:34
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:29:34
# Copyright (c) FederalLab. All rights reserved.
import json

import matplotlib.pyplot as plt
import numpy as np


def plot(files, labels, attributes='accuracy', mode='train'):
    if isinstance(files, str):
        files = [
            files,
        ]
    if isinstance(labels, str):
        labels = [
            labels,
        ]

    assert mode in ['train', 'test'], 'Invalid mode.'

    xs, ys = [], []
    for file in files:
        x, y = [], []
        with open(file, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                if data['mode'] == mode:
                    x.append(data['version'])
                    y.append(data[attributes])
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)

    plt.figure()
    for x, y, l in zip(xs, ys, labels):
        plt.plot(x, y, label=l)
    plt.title(f'{mode} {attributes} curve')
    plt.legend(
        loc='lower right' if attributes == 'accuracy' else 'upper right')
    plt.show()
