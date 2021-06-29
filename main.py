import os
import sys
sys.path.insert(0, "/Users/densechen/code/OpenFed")
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import torch.optim as optim
import openfed
import openfed.aggregate as aggregate
from openfed.unified.step import StepAt
from openfed.utils import time_string
from benchmark.models import LogisticRegression
from benchmark.datasets import get_mnist


# >>> Import OpenFed

# >>> set log level
openfed.logger.log_level(level="INFO")

# >>> Get default arguments from OpenFed
args = openfed.parser.parse_args()

epochs = 5

# >>> Specify an API for building federated learning
openfed_api = openfed.API(frontend=args.rank > 0)

# >>> Specify a aggregate trigger
# It means that every 10 received models will make an aggregate operation.
aggregate_trigger = openfed.AggregateCount(
    count=1, checkpoint="/tmp/openfed-model")

# >>> Set the aggregate trigger
openfed_api.set_aggregate_triggers(aggregate_trigger)

# >>> Register more step functions.
# You can register a step function to openfed_api like following:
# stop_at_version = openfed.StopAtVersion(max_version=10)
# openfed_api.register_step(stop_at_version)
# Or use the with context to add a sequence of step function to openfed_api automatically.
with StepAt(openfed_api):
    # In other means, it will train for epochs.
    openfed.StopAtVersion(max_version=epochs)

# >>> Connect to Address.
openfed_api.build_connection(address=openfed.Address(args=args))

# Build Network
net = LogisticRegression(784, 10)

# Define optimizer (use the same optimizer in both server and client)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Define aggregator (actually, this is only used for server end): FedAvg, ElasticAggregator
aggregator = aggregate.AverageAggregator(net.parameters())

# >>> Set optimizer and aggregator for federated learning.
openfed_api.set_aggregator_and_optimizer(aggregator, optimizer)

# >>> Tell OpenFed API which data should be transferred.
openfed_api.set_state_dict(net.state_dict(keep_vars=True))

# Load dataset and create dataloader
train_dataset = get_mnist('data', total_parts=1000, train=True)
test_dataset = MNIST('data', train=False, transform=ToTensor())
train_dataloader = DataLoader(
    train_dataset, batch_size=12, shuffle=True, num_workers=2)
test_dataloader = DataLoader(
    test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# Context `with openfed_api` will go into the specified settings about openfed_api.
# Otherwise, will use the default one which shared by global OpenFed world.
with openfed_api:

    # >>> If openfed_api is a backend, call `run()` will go into the loop ring.
    # >>> Call `start()` will run it as a thread.
    # >>> If openfed_api is a frontend, call `run()` will directly skip this function automatically.
    openfed_api.run()

    for i in range(epochs):
        print(f"{time_string()}: Simulation @{i}")
        # Download latest model.
        print(f"{time_string()}: Downloading latest model from server.")
        if not openfed_api.download():
            print(f"Downloading failed.")
            break
        # Downloaded
        print(f"{time_string()}: Downloaded!")

        train_dataset.set_part_id(random.randint(0, train_dataset.total_parts))
        process = tqdm(train_dataloader)
        for data in process:
            input, target = data

            # Start a standard forward/backward pass.
            optimizer.zero_grad()
            output = net(input)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()
            process.set_description(f"loss: {loss:.2f}")

        # Upload trained model
        print(f"{time_string()}: Uploading trained model to server.")
        if not openfed_api.upload():
            print("Uploading failed.")
            break
        print(f"{time_string()}: Uploaded!")

        # >>> Update inner model version
        openfed_api.update_version()

# >>> Finished
openfed_api.finish()

print(f"Finished.\nExit Client @{openfed_api.nick_name}.")
