import time

import openfed
from torch.utils.data import random_split


class Trainer(object):
    def __init__(self, openfed_api, model, optimizer, dataloader):
        self.openfed_api = openfed_api
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

        self.task_info = openfed.TaskInfo()

    def train_epoch(self, epoch=1):
        """Train model for several epochs. 
        Returns:
            average training accuracy
            average training loss
            duration: time in seconds
        """
        self.model.train()
        accuracies = []
        losses = []
        tic = time.time()
        for e in range(epoch):
            for data in self.dataloader:
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.model.loss_fn(output, y)
                acc = self.model.acc_fn(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accuracies.append(acc.item())
                losses.append(loss.item())
        toc = time.time()
        return sum(accuracies)/len(accuracies), sum(losses)/len(losses), toc-tic

    def acg_epoch(self, max_samples=-1):
        """Do acg step to accumulate gradient for some federated optimizer.
        Returns:
            duration: time in seconds
        """
        dataset = self.dataloader.dataset

        if max_samples > 0:
            acg_subdataset, _ = random_split(dataset, [max_samples, -1])
        else:
            acg_subdataset = dataset

        self.dataloader.dataset = acg_subdataset
        tic = time.time()
        self.optimizer.acg(self.model, self.dataloader,
                           loss_fn=self.model.loss_fn, device=self.device)
        toc = time.time()

        self.dataloader.dataset = dataset

        return toc - tic

    @openfed.api.device_offline_care
    def finish_training(self, **kwargs):
        """
        Args: 
            kwargs: extra  information added to task info.
        """
        self.optimizer.round()

        self.task_info.update(kwargs)

        if not self.openfed_api.transfer(to=True, task_info=self.task_info):
            return False
        else:
            self.optimizer.clear_buffer()
            return True

    @openfed.api.device_offline_care
    def start_training(self):
        if not self.openfed_api.transfer(to=False, task_info=self.task_info):
            return False
        else:
            part_id = self.task_info.part_id  # type: ignore
            self.dataloader.set_part_id(part_id)
            return True
