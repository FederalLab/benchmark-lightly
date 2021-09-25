# @Author            : FederalLab
# @Date              : 2021-09-26 00:28:50
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:28:50
# Copyright (c) FederalLab. All rights reserved.
def top_one_acc(target, predict):
    # Start a standard forward inference pass.
    predict = predict.max(1, keepdim=True)[1]
    return predict.eq(target.view_as(predict)).sum() / len(target)
