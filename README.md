# A high speed simulation benchmark of federated learning

This simulation benchmark can do federated learning within a single node with/without GPU.

## Setup

### Install

```bash
git clone git@github.com:FederalLab/benchmark-lightly.git
cd benchmark-lightly
pip install -r requirements.txt
```

### Download dataset

`md5sum` is required in your system path.
Refer to `datasets/xxx/README.md` for more details about preparing dataset.

## Experiment

Refer to `docs/xxx.ipynb` for more details about different experiments settings and results.


## Fast Test

```bash
# Download mnist
python benchmark/datasets/mnist/mnist.py benchmark/datasets/mnist/data

# Do a simple test
rm -rf /tmp/mnist.share
python -m openfed.tools.launch --nproc_per_node 6  --logdir /tmp benchmark/run.py --fed_init_method file:///tmp/mnist.share --network_args input_dim:784 --samples 10 --gpu
```