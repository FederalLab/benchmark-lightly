# benchmark

## Install

```bash
conda activate openfed

# Clone the latest version of OpenFed, and install it
python setup.py install
```

## Run

1. Run `python benchmark/datasets/mnist.py` to download MNIST dataset.
2. Run `python -m openfed.tools.launch --nproc_per_node 6 --logdir /tmp main.py --fed_init_method file:///tmp/openfed.sharefile` to start a simulation with 6 node (1 server, 5 client). (Make sure `/tmp/openfed.sharefile` is not existed.)

## Dataset

```bash
cd data && bash download.sh && cd ..
```