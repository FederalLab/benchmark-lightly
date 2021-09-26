import json
from typing import Any, Dict, List

from openfed.functional import meta_reduce


def meta_reduce_log(meta_list: List[Dict[str, Any]], log_dir: str = None):
    reduce_keys = ['loss', 'accuracy']
    reduce_meta = meta_reduce(meta_list, reduce_keys)

    reduce_meta['mode'] = meta_list[0]['mode']
    reduce_meta['version'] = meta_list[0]['version']

    if log_dir is not None:
        with open(log_dir, 'a') as f:
            json.dump(reduce_meta, f, )
            f.write('\n')
    return f"loss: {reduce_meta['loss']:.2f}, "\
        f"accuracy: {reduce_meta['accuracy']:.2f}"
