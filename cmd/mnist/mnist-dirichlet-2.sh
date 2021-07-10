TAIL=mnist-dirichlet-2
rm -rf /tmp/openfed.sharefile.$TAIL

python -m openfed.tools.launch\
    --nproc_per_node 11\
    --logdir /tmp\
    $BENCHMARK_ROOT/benchmark/main.py\
    --fed_init_method file:///tmp/openfed.sharefile.$TAIL\
    --dataset mnist\
    --data_root $BENCHMARK_ROOT/data\
    --partition dirichlet\
    --num_parts 100\
    --epochs 1\
    --rounds 100\
    --samples 10\
    --test_samples 10\
    --ft_optim sgd\
    --bk_optim sgd\
    --pipe none\
    --agg naive\
    --ft_lr 0.01\
    --bk_lr 1.0\
    --ft_lr_sch none\
    --bk_lr_sch none\
    --bz 10\
    --gpu\
    --log_level SUCCESS\
    --ckpt /tmp/openfed.$TAIL\
    --task cls\
    --network lr\
    --network_cfg input_dim:784 output_dim:10\
    --log_file $BENCHMARK_ROOT/logs/$TAIL.json\
    --seed 0
