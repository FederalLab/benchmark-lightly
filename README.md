# benchmark

## Install

```bash
conda activate openfed

# Clone the latest version of OpenFed, and install it
python setup.py install
```

## Run

```bash
python benchmark/datasets/mnist.py

rm -rf /tmp/openfed*

python -m openfed.tools.launch\
    --nproc_per_node 6\
    --logdir /tmp\
    main.py\
    --fed_init_method file:///tmp/openfed.sharefile\
    --network_cfg\
    input_dim:784\
    output_dim:10\
    --samples\
    10\
    --test_samples\
    20\
    --pipe\
    prox
```

## Dataset

```bash
cd data && bash download.sh && cd ..
```