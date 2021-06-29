# Dataset

It is recommended to symlink the dataset root to `$BENCHMARK/data`.
If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```bash
data
├── download.sh
├── Federated_CIFAR100_TFF
│   ├── fed_cifar100_test.h5
│   └── fed_cifar100_train.h5
├── Federated_EMNIST_TFF
│   ├── fed_emnist_test.h5
│   └── fed_emnist_train.h5
├── Federated_Shakespeare_FedProx
│   ├── all_data_niid_2_keep_0_test_8.json
│   └── all_data_niid_2_keep_0_train_8.json
├── Federated_Shakespeare_TFF
│   ├── shakespeare_test.h5
│   └── shakespeare_train.h5
├── Federated_StackOverFlow_TFF
    ├── stackoverflow_held_out.h5
    ├── stackoverflow_nwp.pkl
    ├── stackoverflow.tag_count
    ├── stackoverflow_test.h5
    ├── stackoverflow_train.h5
    └── stackoverflow.word_count
```

## Download

Run `bash download.sh` to download all datasets.

## Valid

```bash
cd $BENCHMARK
python benchmark/datasets/cifar100.py
+-------+---------+--------+------+
| Parts | Samples |  Mean  | Var  |
+-------+---------+--------+------+
|  500  |  50000  | 100.00 | 0.00 |
+-------+---------+--------+------+

python benchmark/datasets/emnist.py
+-------+---------+--------+---------+
| Parts | Samples |  Mean  |   Var   |
+-------+---------+--------+---------+
|  3400 |  671585 | 197.53 | 5879.93 |
+-------+---------+--------+---------+

python benchmark/datasets/shapespeare.py
+-------+---------+---------+-------------+
| Parts | Samples |   Mean  |     Var     |
+-------+---------+---------+-------------+
|  143  |  413629 | 2892.51 | 29667071.69 |
+-------+---------+---------+-------------+
+-------+---------+-------+---------+
| Parts | Samples |  Mean |   Var   |
+-------+---------+-------+---------+
|  715  |  16068  | 22.47 | 1240.68 |
+-------+---------+-------+---------+
```