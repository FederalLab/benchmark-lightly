TAIL=mnist-iid-3
rm -rf /tmp/openfed.sharefile.$TAIL

python -m openfed.tools.launch\
    --nproc_per_node 11\
    --logdir /tmp\
    $BENCHMARK_ROOT/benchmark/main.py\
    --fed_init_method file:///tmp/openfed.sharefile.$TAIL\
    --dataset mnist\
    --data_root $BENCHMARK_ROOT/data\
    --partition iid\
    --num_parts 100\
    --epochs 1\
    --rounds 100\
    --samples 10\
    --test_samples 10\
    --follower_optim sgd\
    --leader_optim sgd\
    --penal none\
    --agg elastic\
    --follower_lr 0.01\
    --leader_lr 1.0\
    --follower_lr_sch none\
    --leader_lr_sch none\
    --bz 10\
    --gpu\
    --log_level SUCCESS\
    --ckpt /tmp/openfed.$TAIL\
    --task cls\
    --network lr\
    --network_cfg input_dim:784 output_dim:10\
    --log_file $BENCHMARK_ROOT/logs/$TAIL.json\
    --seed 0
