rm -rf /tmp/openfed.sharefile.emnist-agg-1

python -m openfed.tools.launch\
    --nproc_per_node 41\
    --logdir /tmp\
    $BENCHMARK_ROOT/benchmark/main.py\
    --fed_init_method file:///tmp/openfed.sharefile.emnist-agg-1\
    --dataset emnist\
    --data_root $BENCHMARK_ROOT/data/Federated_EMNIST_TFF\
    --epochs 1\
    --rounds 100\
    --samples 100\
    --test_samples 100\
    --follower_optim sgd\
    --leader_optim sgd\
    --penal none\
    --agg average\
    --follower_lr 0.001\
    --leader_lr 1.0\
    --follower_lr_sch none\
    --leader_lr_sch none\
    --bz 10\
    --gpu\
    --log_level SUCCESS\
    --ckpt /tmp/openfed.emnist-agg-1\
    --task cls\
    --network emnist\
    --network_cfg only_digits:False\
    --log_file $BENCHMARK_ROOT/logs/emnist-agg-1.json\
    --seed 0
