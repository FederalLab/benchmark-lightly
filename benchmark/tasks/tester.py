import time


class Tester(object):
    def __init__(self, openfed_api, model, dataloader):
        self.openfed_api = openfed_api
        self.model = model

        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

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

    def finish_testing(self, task_info):
        self.openfed_api.update_version(task_info.version)
        if not self.openfed_api.transfer(to=True, task_info=task_info):
            return False
        else:
            return True

    def start_testing(self, task_info):
        part_id = task_info.part_id  # type: ignore
        self.dataloader.set_part_id(part_id)
