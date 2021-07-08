from typing import List, Union

from openfed.common import TaskInfo
from openfed.container import AutoReducer

from tensorboardX import SummaryWriter


class AutoReducerTb(AutoReducer):
    def __init__(self,
                 weight_key: str = None,
                 reduce_keys: Union[str, List[str]] = None,
                 additional_keys: Union[str, List[str]] = None,
                 log_dir: str = '/tmp'):
        super().__init__(weight_key, reduce_keys, additional_keys)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_test_accuracy = 0.0
        self.best_train_accuracy = 0.0

    def reduce(self) -> TaskInfo:
        reduced_task_info = super().reduce()

        # Log to wandb
        if reduced_task_info.train:
            self.writer.add_scalars(
                'train',
                {
                    "accuracy": reduced_task_info.accuracy,
                    "train": reduced_task_info.loss,
                },
                reduced_task_info.version,
            )
            if reduced_task_info.accuracy > self.best_train_accuracy:
                self.best_train_accuracy = reduced_task_info.accuracy
        else:
            self.writer.add_scalars(
                'test',
                {
                    "accuracy": reduced_task_info.accuracy,
                    "train": reduced_task_info.loss,
                },
                reduced_task_info.version,
            )
            if reduced_task_info.accuracy > self.best_test_accuracy:
                self.best_test_accuracy = reduced_task_info.accuracy

        return reduced_task_info
