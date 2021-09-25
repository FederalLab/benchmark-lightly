# @Author            : FederalLab
# @Date              : 2021-09-26 00:32:20
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:32:20
# Copyright (c) FederalLab. All rights reserved.
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
