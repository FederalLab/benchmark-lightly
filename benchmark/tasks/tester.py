# @Author            : FederalLab
# @Date              : 2021-09-26 00:29:23
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:29:23
# Copyright (c) FederalLab. All rights reserved.
import time

import torch


class Tester(object):
    def __init__(self, maintainer, model, dataloader):
        self.maintainer = maintainer
        self.model = model

        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def test_epoch(self):
        """Train model for several epochs.

        Returns:
            average training accuracy
            average training loss
            duration: time in seconds
        """
        self.model.eval()
        accuracies = []
        losses = []
        tic = time.time()
        for data in self.dataloader:
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.model.loss_fn(output, y)
            acc = self.model.accuracy_fn(y, output)
            accuracies.append(acc.item())
            losses.append(loss.item())
        toc = time.time()
        return sum(accuracies) / len(accuracies), sum(losses) / len(
            losses), toc - tic

    def finish_testing(self, task_info):
        self.maintainer.update_version(task_info.version)
        if not self.maintainer.transfer(to=True, task_info=task_info):
            return False
        else:
            return True

    def start_testing(self, task_info):
        part_id = task_info.part_id  # type: ignore
        self.dataloader.dataset.set_part_id(part_id)
