# @Author            : FederalLab
# @Date              : 2021-09-26 00:29:26
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:29:26
# Copyright (c) FederalLab. All rights reserved.
import os
import time
import warnings

import torch


class Trainer(object):
    def __init__(self,
                 maintainer,
                 model,
                 optimizer,
                 dataloader,
                 cache_folder: str = '/tmp'):
        self.maintainer = maintainer
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device
        self.cache_folder = cache_folder
        # Remove cache folder
        if os.path.exists(self.cache_folder):
            warnings.warn(f'The cache folder ({cache_folder}) already exists.')
        os.makedirs(self.cache_folder, exist_ok=True)

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
                acc = self.model.accuracy_fn(y, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accuracies.append(acc.item())
                losses.append(loss.item())
        toc = time.time()
        return sum(accuracies) / len(accuracies), sum(losses) / len(
            losses), toc - tic

    def acg_epoch(self, max_acg_step=-1):
        """Do acg step to accumulate gradient for some federated optimizer.

        Returns:
            duration: time in seconds
        """
        tic = time.time()
        self.optimizer.max_acg_step = max_acg_step

        self.optimizer.acg(self.model, self.dataloader)
        toc = time.time()
        return toc - tic

    def finish_training(self, task_info):
        """
        Args:
            kwargs: extra  information added to task info.
        """
        self.optimizer.round()
        self.maintainer.update_version(task_info.version)

        part_id = task_info.part_id
        # Save inner state of self.optimizer
        cache_file = os.path.join(self.cache_folder, f'{part_id}.pth')
        torch.save(self.optimizer.state_dict(), cache_file)


        self.maintainer.package(self.optimizer)
        self.maintainer.step(download=False, meta=task_info)

        self.optimizer.clear_state_dict()


    def start_training(self, task_info):
        part_id = task_info.part_id  # type: ignore
        self.dataloader.dataset.set_part_id(part_id)

        # Load inner state of self.optimizer
        # This step is vital for that some optimizers, such as fedscaffold
        # need the previous state to continue training.
        cache_file = os.path.join(self.cache_folder, f'{part_id}.pth')
        if os.path.exists(cache_file):
            self.optimizer.load_state_dict(torch.load(cache_file))
