import time

class Trainer(object):
    def __init__(self, openfed_api, model, optimizer, dataloader):
        self.openfed_api = openfed_api
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

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
        return sum(accuracies)/len(accuracies), sum(losses)/len(losses), toc-tic

    def acg_epoch(self, max_acg_step=-1):
        """Do acg step to accumulate gradient for some federated optimizer.
        Returns:
            duration: time in seconds
        """
        tic = time.time()
        self.optimizer.max_acg_step = max_acg_step

        self.optimizer.acg(self.model, self.dataloader,
                           loss_fn=self.model.loss_fn, device=self.device)
        toc = time.time()
        return toc - tic

    def finish_training(self, task_info):
        """
        Args: 
            kwargs: extra  information added to task info.
        """
        self.optimizer.round()
        self.openfed_api.update_version(task_info.version)
        if not self.openfed_api.transfer(to=True, task_info=task_info):
            return False
        else:
            self.optimizer.clear_buffer()
            return True

    def start_training(self, task_info):
        part_id = task_info.part_id  # type: ignore
        self.dataloader.dataset.set_part_id(part_id)
