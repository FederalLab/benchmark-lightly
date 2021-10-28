# Federated Learning Simulator on Light Tasks

We include several most common used federated tasks in this benchmark. You can validate and compare different federated algorithms under various experiment settings efficiently.

## Install

```bash
git clone git@github.com:FederalLab/benchmark-lightly.git
cd benchmark-lightly
pip install -r requirements.txt
python setup.py install
```

## Benchmark

Refer to `docs/` for more details.
We provide a flexible settings configuration for benchmark, rather than given an specified settings, since different settings will have a large impact on performance. It is more convenient to validate different algorithms in there ideal situation, such as non-iid or iid distribution, full-activated or partially activated clients.

## Citation

If you find this project useful in your research, please consider cite:

```
@misc{chen2021openfed,
      title={OpenFed: A Comprehensive and Versatile Open-Source Federated Learning Framework},
      author={Dengsheng Chen and Vince Tan and Zhilin Lu and Jie Hu},
      year={2021},
      eprint={2109.07852},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
