# CIFAR100

Loads a federated version of the CIFAR-100 dataset.
The dataset is downloaded and cached locally. If previously downloaded, it
tries to load the dataset from cache.
The dataset is derived from the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The training and testing examples are partitioned across 500 and 100 clients (respectively).
No clients share any data samples, so it is a true partition of CIFAR-100.
The train clients have string client IDs in the range [0-499], while the test
clients have string client IDs in the range [0-99]. The train clients form a
true partition of the CIFAR-100 training split, while the test clients form a
true partition of the CIFAR-100 testing split.
The data partitioning is done using a hierarchical Latent Dirichlet Allocation
(LDA) process, referred to as the [Pachinko Allocation Method](https://people.cs.umass.edu/~mccallum/papers/pam-icml06.pdf) (PAM).
This method uses a two-stage LDA process, where each client has an associated
multinomial distribution over the coarse labels of CIFAR-100, and a
coarse-to-fine label multinomial distribution for that coarse label over the
labels under that coarse label. The coarse label multinomial is drawn from a
symmetric Dirichlet with parameter 0.1, and each coarse-to-fine multinomial
distribution is drawn from a symmetric Dirichlet with parameter 10. Each
client has 100 samples. To generate a sample for the client, we first select
a coarse label by drawing from the coarse label multinomial distribution, and
then draw a fine label using the coarse-to-fine multinomial distribution. We
then randomly draw a sample from CIFAR-100 with that label (without
replacement). If this exhausts the set of samples with this label, we
remove the label from the coarse-to-fine multinomial and re-normalize the
multinomial distribution.
Data set sizes:

- train: 50,000 examples
- test: 10,000 examples

## Download

```bash
cd data/cifar100

mkdir -pv data/raw && cd data/raw && wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/fed_cifar100.tar.bz2

tar -xvf fed_cifar100.tar.bz2 && rm fed_cifar100.tar.bz2
```
