# benchmark

## Install

```bash
conda activate openfed

# Clone the latest version of OpenFed, and install it
python setup.py install
```

## Run

1. Run `python benchmark/datasets/mnist.py` to download MNIST dataset.
2. Run `python -m openfed.tools.launch --nproc_per_node 6 --logdir /tmp --server_output main.py --init_method file:///tmp/openfed.sharefile` to start a simulation with 6 node (1 server, 5 client).
3. Refer to `log/` for more output details.

## Dataset

```bash
cd data && bash download.sh && cd ..
```