# MNIST

## Download

```bash
mkdir -pv data/raw
python mnist.py data/raw
```

## Test

```bash
python benchmark/datasets/mnist/mnist.py benchmark/datasets/mnist/data
+-------+---------+--------+------+
| Parts | Samples |  Mean  | Var  |
+-------+---------+--------+------+
|  100  |  59600  | 596.00 | 0.00 |
+-------+---------+--------+------+
```
