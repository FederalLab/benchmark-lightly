import json
import numpy as np
import matplotlib.pyplot as plt


def parser_json(lines):
    return [json.loads(line+'}') for line in lines.split('}') if len(line) > 0]


def plot(files, labels, attributes="accuracy", train=True):
    if isinstance(files, str):
        files = [files]
    if isinstance(labels, str):
        labels = [labels]

    xs, ys = [], []
    for file in files:
        x, y = [], []
        with open(file, 'r') as f:
            data = parser_json(f.read())
            for d in data:
                if d['train'] == train:
                    x.append(d['version'])
                    y.append(d[attributes])
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)

    plt.figure()
    for x, y, l in zip(xs, ys, labels):
        plt.plot(x, y, label=l)

    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    plot('logs/celeba/default/celeba.json', 'default')
