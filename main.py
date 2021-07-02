import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed')

import openfed
import openfed.api as of_api
import torch.nn.functional as F
import torch.optim as optim
from openfed.container import AutoReducer, AverageAggregator
from openfed.core.inform import LRTracker
from openfed.utils import time_string
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from benchmark.datasets import get_mnist
from benchmark.models import LogisticRegression

# >>> set log level
openfed.logger.log_level(level="INFO")

# >>> Get default arguments from OpenFed
args = openfed.parser.parse_args()
openfed.logger.log_to_file(f"log/{args.rank}.txt")

epochs = 30
total_parts = 100
samples = 15

# >>> Specify an API for building federated learning
openfed_api = openfed.API(frontend=args.rank > 0)

# Build Network
net = LogisticRegression(784, 10)

# Define optimizer (use the same optimizer in both server and client)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Define aggregator (actually, this is only used for server end): FedAvg, ElasticAggregator
aggregator = AverageAggregator(net.parameters())

# >>> Add an auto-reducer to compute task info
auto_reducer = AutoReducer(weight_key='instances')
aggregator.register_reducer(auto_reducer)

# >>> Set optimizer and aggregator for federated learning.
openfed_api.set_aggregator_and_optimizer(aggregator, optimizer)

# Define a LRScheduler
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs,)

# >>> Synchronize LR Scheduler between Frontend and Backend
lr_tracker = LRTracker(lr_scheduler)
openfed_api.add_informer_hook(lr_tracker)

# >>> Tell OpenFed API which data should be transferred.
openfed_api.set_state_dict(net.state_dict(keep_vars=True))

# >>> Register more step functions.
# You can register a step function to openfed_api like following:
# stop_at_version = openfed.StopAtVersion(max_version=10)
# openfed_api.register_step(stop_at_version)
# Or use the with context to add a sequence of step function to openfed_api automatically.
with of_api.StepAt(openfed_api):
    of_api.AggregateCount(
        count=samples, checkpoint="/tmp/openfed-model", lr_scheduler=lr_scheduler)
    of_api.AfterDownload()
    of_api.Dispatch(total_parts=total_parts, samples=samples)

    of_api.StopAtVersion(max_version=epochs)
# >>> Connect to Address.
openfed_api.build_connection(address=openfed.Address(args=args))

# Load dataset and create dataloader
train_dataset = get_mnist('data', total_parts=total_parts, train=True)
test_dataset = MNIST('data', train=False, transform=ToTensor())
train_dataloader = DataLoader(
    train_dataset, batch_size=10, shuffle=True, num_workers=0)
test_dataloader = DataLoader(
    test_dataset, batch_size=1000, shuffle=False, num_workers=0)

# Context `with openfed_api` will go into the specified settings about openfed_api.
# Otherwise, will use the default one which shared by global OpenFed world.
with openfed_api:

    # >>> If openfed_api is a backend, call `run()` will go into the loop ring.
    # >>> Call `start()` will run it as a thread.
    # >>> If openfed_api is a frontend, call `run()` will directly skip this function automatically.
    openfed_api.backend_loop()

    while True:
        # Download latest model.
        print(f"{time_string()}: Downloading latest model from server.")
        if not openfed_api.download():
            print(f"Downloading failed.")
            break

        # Downloaded
        print(f"{time_string()}: Downloaded!")

        task_info = openfed_api.get_task_info()
        part_id, version = task_info.get(
            'part_id'), task_info.get('version')

        print(f"{time_string()}")
        print(task_info)

        train_dataset.set_part_id(part_id)

        # Train
        total_loss = []
        for data in train_dataloader:
            input, target = data

            # Start a standard forward/backward pass.
            optimizer.zero_grad()
            output = net(input)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()
            total_loss.append(loss.item())
        total_loss = sum(total_loss)/len(total_loss)

        # Test
        correct = []
        for data in test_dataloader:
            input, target = data

            # Start a standard forward inference pass.
            predict = net(input).max(1, keepdim=True)[1]
            correct.append(predict.eq(target.view_as(
                predict)).sum().item() / len(target))
        accuracy = sum(correct) / len(correct)

        task_info.set("instances", len(train_dataloader.dataset))
        task_info.set("loss", total_loss)
        task_info.set("accuracy", accuracy)
        task_info.set("version", version + 1)

        # Set task info
        openfed_api.set_task_info(task_info)
        openfed_api.update_version(version + 1)

        # Upload trained model
        print(f"{time_string()}: Uploading trained model to server.")
        if not openfed_api.upload():
            print("Uploading failed.")
            break
        print(f"{time_string()}: Uploaded!")

# >>> Finished
openfed_api.finish()

print(f"Finished.\nExit Client @{openfed_api.nick_name}.")
