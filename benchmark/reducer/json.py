# type: ignore

import json
from typing import List, Union

from openfed.common import TaskInfo
from openfed.container import AutoReducer


class AutoReducerJson(AutoReducer):
    def __init__(self,
                 weight_key: str = None,
                 reduce_keys: Union[str, List[str]] = None,
                 additional_keys: Union[str, List[str]] = None,
                 log_file: str = None):
        super().__init__(weight_key, reduce_keys, additional_keys)

        self.log_file = log_file
        self.best_test_accuracy = 0.0
        self.best_train_accuracy = 0.0

        self.task_info_list = []

    def reduce(self) -> TaskInfo:
        reduced_task_info = super().reduce()

        if reduced_task_info.train:
            if reduced_task_info.accuracy > self.best_train_accuracy:
                self.best_train_accuracy = reduced_task_info.accuracy
        else:
            if reduced_task_info.accuracy > self.best_test_accuracy:
                self.best_test_accuracy = reduced_task_info.accuracy
        if self.log_file is not None:
            with open(self.log_file, 'w') as f:
                reduced_task_info["best_train_accuracy"] = self.best_train_accuracy
                reduced_task_info["best_test_accuracy"] = self.best_test_accuracy
                self.task_info_list.append(reduced_task_info)
                json.dump(self.task_info_list, f)

        return reduced_task_info
