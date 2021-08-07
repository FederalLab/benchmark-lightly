import time

import openfed


class Tester(object):
    def __init__(self, openfed_api, model, dataloader):
        self.openfed_api = openfed_api
        self.model = model

        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

        self.task_info = openfed.TaskInfo()

    def test_epoch(self):
        """Train model for several epochs. 
        Returns:
            average training accuracy
            average training loss
            duration: time in seconds
        """
        self.model.test()
        accuracies = []
        losses = []
        tic = time.time()
        for data in self.dataloader:
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.model.loss_fn(output, y)
            acc = self.model.acc_fn(output, y)
            accuracies.append(acc.item())
            losses.append(loss.item())
        toc = time.time()
        return sum(accuracies)/len(accuracies), sum(losses)/len(losses), toc-tic

    @openfed.api.device_offline_care
    def finish_testing(self, **kwargs):
        """
        Args: 
            kwargs: extra  information added to task info.
        """
        self.task_info.update(kwargs)

        if not self.openfed_api.transfer(to=True, task_info=self.task_info):
            return False
        else:
            return True

    @openfed.api.device_offline_care
    def start_testing(self):
        if not self.openfed_api.transfer(to=False, task_info=self.task_info):
            return False
        else:
            part_id = self.task_info.part_id  # type: ignore
            self.dataloader.set_part_id(part_id)
            return True
