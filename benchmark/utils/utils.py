# @Author            : FederalLab
# @Date              : 2021-09-26 00:29:37
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:29:37
# Copyright (c) FederalLab. All rights reserved.
import argparse


class StoreDict(argparse.Action):
    """Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings,
                                        dest,
                                        nargs=nargs,
                                        **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(':')[0]
            value = ':'.join(arguments.split(':')[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)
