import json
import numpy as np
import matplotlib.pyplot as plt


def parser_json(lines):
    return [json.loads(line+'}') for line in lines.split('}') if len(line) > 0]


def plot(files, labels, attributes="accuracy", mode='train'):
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
                if d['mode'] == mode:
                    x.append(d['version'])
                    y.append(d[attributes])
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)

    plt.figure()
    for x, y, l in zip(xs, ys, labels):
        plt.plot(x, y, label=l)
    plt.title(f'{mode} {attributes} curve')
    plt.legend(loc="lower right" if attributes == 'accuracy' else "upper right")
    plt.show()


if __name__ == '__main__':
    plot('logs/celeba/default/celeba.json', 'default')
