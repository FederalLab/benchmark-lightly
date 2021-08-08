# type: ignore

import json
from typing import List, Union

from openfed.common import TaskInfo
from openfed.container import AutoReducer


class AutoReducerJson(AutoReducer):
    def __init__(self,
                 reduce_keys: Union[str, List[str]] = None,
                 weight_key: str = None,
                 ignore_keys: List[str] = None,
                 log_file: str = None):
        super().__init__(reduce_keys, weight_key, ignore_keys)

        self.log_file = log_file

    def reduce(self) -> TaskInfo:
        r_task_info = super().reduce()

        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                json.dump(r_task_info, f)

        return r_task_info
