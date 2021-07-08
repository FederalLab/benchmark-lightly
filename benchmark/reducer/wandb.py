from typing import List, Union

from openfed.common import TaskInfo
from openfed.container import AutoReducer

import wandb


class AutoReducerWanb(AutoReducer):
    def __init__(self,
                 weight_key: str = None,
                 reduce_keys: Union[str, List[str]] = None,
                 additional_keys: Union[str, List[str]] = None,
                 project: str = 'OpenFed-benchmark'):
        super().__init__(weight_key, reduce_keys, additional_keys)
        wandb.init(project=project)

        self.project = project
        self.best_test_accuracy = 0.0
        self.best_train_accuracy = 0.0

    def reduce(self) -> TaskInfo:
        reduced_task_info = super().reduce()

        # Log to wandb
        if reduced_task_info.train:
            wandb.log(
                {"train/version": reduced_task_info.version,
                 "train/accuracy": reduced_task_info.accuracy,
                 "train/loss": reduced_task_info.loss,
                 }
            )
            if reduced_task_info.accuracy > self.best_train_accuracy:
                self.best_train_accuracy = reduced_task_info.accuracy
        else:
            wandb.log(
                {"test/version": reduced_task_info.version,
                 "test/accuracy": reduced_task_info.accuracy,
                 "test/loss": reduced_task_info.loss,
                 }
            )
            if reduced_task_info.accuracy > self.best_test_accuracy:
                self.best_test_accuracy = reduced_task_info.accuracy

        return reduced_task_info
