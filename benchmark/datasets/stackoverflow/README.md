# StackOverFlow

## Download

```bash
# Federated StackOverFlow TFF
mkdir -pv data/raw && cd data/raw
wget --no-check-certificate --no-proxy  https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.tag_count.tar.bz2
wget --no-check-certificate --no-proxy  https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.word_count.tar.bz2
wget --no-check-certificate --no-proxy  https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.tar.bz2
wget --no-check-certificate --no-proxy  https://fedml.s3-us-west-1.amazonaws.com/stackoverflow_nwp.pkl

tar -xvf stackoverflow.tag_count.tar.bz2 && rm -rf stackoverflow.tag_count.tar.bz2
tar -xvf stackoverflow.word_count.tar.bz2 && rm -rf stackoverflow.word_count.tar.bz2
tar -xvf stackoverflow.tar.bz2 && rm -rf stackoverflow.tar.bz2
```

## Test

```bash
python benchmark/datasets/stackoverflow/stackoverflow.py
+--------+-----------+--------+------------+
| Parts  |  Samples  |  Mean  |    Var     |
+--------+-----------+--------+------------+
| 342477 | 135818730 | 396.58 | 1635695.82 |
+--------+-----------+--------+------------+
+--------+-----------+--------+------------+
| Parts  |  Samples  |  Mean  |    Var     |
+--------+-----------+--------+------------+
| 342477 | 135818730 | 396.58 | 1635695.82 |
+--------+-----------+--------+------------+
```