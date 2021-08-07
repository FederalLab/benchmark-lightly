# Datasets

## Download

You can refer to subfolder for more details about each dataset, or just download all dataset from [GoogleDriver](), [BaiduYun]() or [Aliyun]().

```bash.
├── README.md
├── __init__.py
├── celeba
│   ├── README.md
│   ├── __init__.py
│   ├── celeba.py
│   ├── data
│   │   ├── all_data
│   │   │   └── all_data.json
│   │   ├── celeba_hist.png
│   │   ├── celeba_hist_nolabel.png
│   │   ├── raw
│   │   │   ├── identity_CelebA.txt
│   │   │   ├── img_align_celeba
│   │   │   └── list_attr_celeba.txt
│   │   ├── rem_user_data
│   │   │   └── all_data_niid_0_keep_5.json
│   │   ├── sampled_data
│   │   │   └── all_data_niid_0.json
│   │   ├── test
│   │   │   └── all_data_niid_0_keep_5_test_9.json
│   │   └── train
│   │       └── all_data_niid_0_keep_5_train_9.json
│   ├── meta
│   │   └── dir-checksum.md5
│   ├── preprocess
│   │   └── metadata_to_json.py
│   ├── preprocess.sh
│   └── stats.sh
├── cifar100
│   ├── README.md
│   ├── __init__.py
│   ├── cifar100.py
│   └── data
│       └── raw
│           ├── fed_cifar100_test.h5
│           └── fed_cifar100_train.h5
├── femnist
│   ├── README.md
│   ├── __init__.py
│   ├── data
│   │   └── raw_data
│   │       ├── by_class
│   │       ├── by_write
│   │       └── by_write.zip
│   ├── femnist.py
│   ├── preprocess
│   │   ├── data_to_json.py
│   │   ├── data_to_json.sh
│   │   ├── get_data.sh
│   │   ├── get_file_dirs.py
│   │   ├── get_hashes.py
│   │   ├── group_by_writer.py
│   │   ├── match_hashes.py
│   │   └── utils.py
│   ├── preprocess.sh
│   └── stats.sh
├── mnist
│   ├── README.md
│   ├── __init__.py
│   ├── data
│   │   └── raw
│   │       └── MNIST
│   └── mnist.py
├── reddit
│   ├── README.md
│   ├── __init__.py
│   ├── build_vocab.py
│   ├── data
│   │   ├── raw
│   │   │   └── RC_2005-12
│   │   ├── reddit_json.zip
│   │   ├── reddit_vocab.pck
│   │   └── �\234�确认\ 912266.crdownload
│   └── preprocess
│       ├── clean_raw.py
│       ├── delete_small_users.py
│       ├── get_json.py
│       ├── get_raw_users.py
│       ├── merge_raw_users.py
│       ├── preprocess.py
│       ├── reddit_utils.py
│       └── run_reddit.sh
├── sent140
│   ├── README.md
│   ├── __init__.py
│   ├── data
│   │   └── raw_data
│   │       └── trainingandtestdata.zip
│   ├── preprocess
│   │   ├── combine_data.py
│   │   ├── data_to_json.py
│   │   ├── data_to_json.sh
│   │   └── get_data.sh
│   ├── preprocess.sh
│   ├── sent140.py
│   └── stats.sh
├── shakespeare
│   ├── README.md
│   ├── __init__.py
│   ├── data
│   │   ├── all_data
│   │   │   └── all_data.json
│   │   ├── raw_data
│   │   │   ├── by_play_and_character
│   │   │   ├── raw_data.txt
│   │   │   └── users_and_plays.json
│   │   ├── rem_user_data
│   │   │   └── all_data_niid_0_keep_0.json
│   │   ├── sampled_data
│   │   │   └── all_data_niid_0.json
│   │   ├── test
│   │   │   └── all_data_niid_0_keep_0_test_9.json
│   │   └── train
│   │       └── all_data_niid_0_keep_0_train_9.json
│   ├── meta
│   │   └── dir-checksum.md5
│   ├── preprocess
│   │   ├── data_to_json.sh
│   │   ├── gen_all_data.py
│   │   ├── get_data.sh
│   │   ├── preprocess_shakespeare.py
│   │   └── shake_utils.py
│   ├── preprocess.sh
│   ├── shakespeare.py
│   └── stats.sh
├── stackoverflow
│   ├── README.md
│   ├── __init__.py
│   ├── data
│   │   └── raw
│   │       ├── stackoverflow.tag_count
│   │       ├── stackoverflow.word_count
│   │       ├── stackoverflow_held_out.h5
│   │       ├── stackoverflow_nwp.pkl
│   │       ├── stackoverflow_test.h5
│   │       └── stackoverflow_train.h5
│   └── stackoverflow.py
├── synthetic
│   ├── README.md
│   ├── __init__.py
│   ├── data
│   │   ├── all_data
│   │   │   └── data.json
│   │   ├── rem_user_data
│   │   │   └── data_niid_0_keep_5.json
│   │   ├── sampled_data
│   │   │   └── data_niid_0.json
│   │   ├── test
│   │   │   └── data_niid_0_keep_5_test_6.json
│   │   └── train
│   │       └── data_niid_0_keep_5_train_6.json
│   ├── data_generator.py
│   ├── main.py
│   ├── meta
│   │   └── dir-checksum.md5
│   ├── preprocess.sh
│   ├── stats.sh
│   └── synthetic.py
└── utils
    ├── __init__.py
    ├── constants.py
    ├── preprocess.sh
    ├── remove_users.py
    ├── sample.py
    ├── sampling_seed.txt
    ├── split_data.py
    ├── split_seed.txt
    ├── stats.py
    └── util.py
```
