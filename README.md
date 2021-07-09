# A high speed simulation benchmark of federated learning

This simulation can do federated learning within a single node with/without GPU.
You can specified any number of process to simulate the federated clients, and we will the process with `fed_rank==0` as the default server. During training time, the server will dispatch task information to each client, and the client will receive the task information and load correctly dataset to train.

## Setup

```bash
# 1. Clone this repo
git clone git@github.com:FederalLab/benchmark.git

# 2. Install dependencies
cd benchmark
pip install -r requirements.txt
python setup.py install

# 3. Download Dataset
cd data && bash download.sh && cd
## Download MNIST
python benchmark/datasets/mnist.py
```

## Run

1. Run multi-processes with `launch`:

```bash
# How many processes do you want to start.
NPROC_PER_NODE=6
LOGDIR=/tmp
# Make sure the file is not existed.
FED_INIT_METHOD=file:///tmp/openfed.sharefile

python -m openfed.tools.launch\
    --nproc_per_node $NPROC_PER_NODE\
    --logdir $LOGDIR\
    benchmark.main\
    --fed_init_method $FED_INIT_METHOD\
    ... # Other parameters
```

2. Launch two processes to debug:

```bash
# Make sure the file is not existed.
FED_INIT_METHOD=file:///tmp/openfed.sharefile

# Terminal 1: Server
python -m benchmark.main --fed_init_method $FED_INIT_METHOD --fed_rank 0 --fed_world_size 2 ... # Other parameters

# Terminal 2: Client
python -m benchmark.main --fed_init_method $FED_INIT_METHOD --fed_rank 1 --fed_world_size 2 ... # Other parameters
```

## Benchmark

```bash
export BENCHMARK_ROOT=`pwd`
```

1. Experiments on EMNIST

|      name      | frontend | backend |   pipe   | aggregator | accuracy |  CMD  |  LOG  |  NOTE  |
| :------------: | :------: | :-----: | :------: | :--------: | :------: | :---: | :---: | :----: |
|  emnist-agg-1  |   sgd    |   sgd   |   none   |  average   |    x     |   x   |   x   | FedSGD |
|  emnist-agg-2  |   sgd    |   sgd   |   none   |   naive    |    x     |   x   |   x   | FedAvg |
|  emnist-agg-3  |   sgd    |   sgd   |   none   |  elastic   |    x     |   x   |   x   |   x    |
|       -        |    -     |    -    |    -     |     -      |    -     |   -   |   -   |   -    |
| emnist-pipe-1  |   sgd    |   sgd   |   prox   |   naive    |    x     |   x   |   x   |   x    |
| emnist-pipe-2  |   sgd    |   sgd   | scaffold |   naive    |    x     |   x   |   x   |   x    |
|       -        |    -     |    -    |    -     |     -      |    -     |   -   |   -   |   -    |
| emnist-optim-1 |   adam   |   sgd   |   none   |   naive    |    x     |   x   |   x   |   x    |
| emnist-optim-2 |   sgd    |  adam   |   none   |   naive    |    x     |   x   |   x   |   x    |
| emnist-optim-3 |   adam   |  adam   |   none   |   naive    |    x     |   x   |   x   |   x    |

1. Experiments on MNIST
