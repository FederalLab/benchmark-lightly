import sys

sys.path.insert(0, '/Users/densechen/code/OpenFed')

from benchmark.models import LogisticRegression
from benchmark.datasets import get_mnist
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from openfed.utils import time_string
from openfed.unified.utils import before_upload
from openfed.unified import StepAt
from openfed.federated.inform import LRTracker
import torch.optim as optim
import torch.nn.functional as F
import openfed.unified as unified
import openfed.aggregate as aggregate
import openfed


# >>> set log level
openfed.logger.log_level(level="DEBUG")

# >>> Get default arguments from OpenFed
args = openfed.parser.parse_args()

epochs = 100
total_parts = 100
samples = 10

# >>> Specify an API for building federated learning
openfed_api = openfed.API(frontend=args.rank > 0)

# >>> Register more step functions.
# You can register a step function to openfed_api like following:
# stop_at_version = openfed.StopAtVersion(max_version=10)
# openfed_api.register_step(stop_at_version)
# Or use the with context to add a sequence of step function to openfed_api automatically.
with StepAt(openfed_api):
    # In other means, it will train for epochs.
    unified.StopAtVersion(max_version=epochs)

# >>> Add a Dispatch
dispatch = unified.Dispatch(total_parts=total_parts, samples=samples)
openfed_api.replace_step(before_upload, dispatch)

# Build Network
net = LogisticRegression(784, 10)

# Define optimizer (use the same optimizer in both server and client)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Define aggregator (actually, this is only used for server end): FedAvg, ElasticAggregator
aggregator = aggregate.AverageAggregator(net.parameters())

# >>> Add an auto-reducer to compute task info
auto_reducer = aggregate.AutoReducer(weight_key='instances')
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

# >>> Specify a aggregate trigger
# It means that every 10 received models will make an aggregate operation.
aggregate_trigger = unified.AggregateCount(
    count=samples, checkpoint="/tmp/openfed-model", lr_scheduler=lr_scheduler)

# >>> Set the aggregate trigger
openfed_api.set_aggregate_triggers(aggregate_trigger)

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
    openfed_api.run()

    while True:
        # Download latest model.
        print(f"{time_string()}: Downloading latest model from server.")
        if not openfed_api.download():
            print(f"Downloading failed.")
            break

        # Downloaded
        print(f"{time_string()}: Downloaded!")

        task_info = openfed_api.get_task_info()
        part_id, version = task_info.get_info(
            'part_id'), task_info.get_info('version')

        print(f"{time_string()}: part_id={part_id}, version={version}")

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

        task_info.add_info("instances", len(train_dataloader.dataset))
        task_info.add_info("loss", total_loss)
        task_info.add_info("accuracy", accuracy)
        task_info.add_info("version", version + 1)

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
