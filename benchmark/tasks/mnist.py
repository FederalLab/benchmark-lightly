# type: ignore
import os

sys.path.insert(0, "/Users/densechen/code/OpenFed")

import argparse
import json
import sys
import time
from glob import glob
from pprint import pprint

import benchmark.datasets as datasets
import benchmark.models as models
import openfed
import torch
import torch.nn.functional as F
from benchmark.datasets.mnist.mnist import get_mnist
from benchmark.reducer import AutoReducerJson
from benchmark.tasks.build_optim import build_optim
from benchmark.tasks.tester import Tester
from benchmark.tasks.trainer import Trainer
from benchmark.utils import StoreDict
from openfed import TaskInfo, time_string
from openfed.core import World, follower, leader
from openfed.data import Analysis
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser("Mnist")

# address
parser.add_argument(
    "--fed_backend",
    default="gloo",
    type=str)
parser.add_argument(
    "--fed_init_method",
    default="tcp://localhost:1994",
    type=str,
    help="opt1: tcp://IP:PORT, opt2: file://PATH_TO_SHAREFILE, opt3:env://")
parser.add_argument(
    "--fed_world_size",
    default=2,
    type=int)
parser.add_argument(
    "--fed_rank",
    "--fed_local_rank",
    default=-1,
    type=int)
parser.add_argument(
    "--fed_group_name",
    default="Admirable",
    type=str,
    help="Add a group name to better recognize each address.")

# dataset
parser.add_argument('--data_root',
                    type=str,
                    default="benchmark/datasets/mnist/data",
                    help='The folder contains all datasets.')
parser.add_argument('--partition',
                    type=str,
                    default='iid',
                    choices=['iid', 'dirichlet', 'power-law'],
                    help='How to split the dataset into different parts.'
                    'Only be used with not federated dataset, such as mnist.'
                    )
parser.add_argument('--num_parts',
                    type=int,
                    default=100,
                    help='The number of the parts to split into.')

# train
parser.add_argument('--epochs',
                    type=int,
                    default=1,
                    help='The epochs trained on local client.')
parser.add_argument('--rounds',
                    type=int,
                    default=10,
                    help='The total rounds for federated training.')
parser.add_argument('--samples',
                    type=int,
                    default=None,
                    help='The number of parts used to train at each round.')
parser.add_argument('--sample_ratio',
                    type=float,
                    default=None,
                    help="The portion of parts used to train at each time, in [0, 1]."
                    "If `samples` set at the same time, this flag will be ignored."
                    )
parser.add_argument('--test_samples',
                    type=int,
                    default=None,
                    help="The number of parts used to test at each round. If not specified, use full test dataset.")
parser.add_argument('--acg_samples',
                    type=int,
                    default=-1,
                    help="The number of samples used to compute acg.")
parser.add_argument('--optim',
                    type=str,
                    default='fedsgd',
                    help='Specify fed optimizer.')
parser.add_argument('--follower_lr',
                    type=float,
                    default=1e-3,
                    help='The learning rate of follower optimizer.')
parser.add_argument('--leader_lr',
                    type=float,
                    default=1.0,
                    help='The learning rate of leader optimizer.')
parser.add_argument('--bz',
                    '--batch_size',
                    type=int,
                    default=10,
                    help='The batch size.')
parser.add_argument('--gpu',
                    action='store_true',
                    default=False,
                    help="Whether to use gpu.")

# log
parser.add_argument('--log_level',
                    type=str,
                    default='SUCCESS',
                    choices=['SUCCESS', 'INFO', 'DEBUG', 'ERROR'],
                    help='The log level of openfed bk.')
parser.add_argument('--log_dir',
                    type=str,
                    default=f'logs/mnist/{time_string()}',
                    help="The dir to log train and test information.")
parser.add_argument('--pretrained',
                    type=str,
                    default='',
                    help='The path to pretrained model.')
parser.add_argument('--ckpt',
                    type=str,
                    default='/tmp/mnist',
                    help='The folder to save checkpoints.')

parser.add_argument('--seed',
                    type=int,
                    default=0,
                    help="Seed for everything.")

args = parser.parse_args()

print('# >>> Set log level...')
openfed.logger.log_level(level=args.log_level)
openfed.utils.seed_everything(args.seed)

args.role = leader if args.fed_rank == 0 else follower

pprint(args.__dict__)

os.makedirs(args.log_dir, exist_ok=True)
with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f)

print("# >>> Dataset...")

if args.partition == 'iid':
    partitioner = openfed.data.IIDPartitioner()
elif args.partition == 'dirichlet':
    partitioner = openfed.data.DirichletPartitioner()
elif args.partition == 'power-law':
    partitioner = openfed.data.PowerLawPartitioner()
else:
    raise NotImplementedError
train_dataset = datasets.get_mnist(
    root=args.data_root, total_parts=args.num_parts, train=True, partitioner=partitioner)
test_dataset = datasets.get_mnist(
    root=args.data_root, total_parts=args.num_parts, train=False, partitioner=partitioner)


print('# >>> Train Dataset:')
Analysis.digest(train_dataset)

print('# >>> Test Dataset:')
Analysis.digest(test_dataset)

print('# >>> DataLoader...')
train_dataloader = DataLoader(
    train_dataset, 
    batch_size  = args.bz,
    shuffle     = True,
    num_workers = 0,
    drop_last   = False)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size  = args.bz,
    shuffle     = False,
    num_workers = 0,
    drop_last   = False)

print('# >>> Device...')
if args.gpu and torch.cuda.is_available():
    args.gpu = args.fed_rank % torch.cuda.device_count()
    args.device = torch.device(args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device('cpu')

print(f"Let's use {args.device}.")

print('# >>> Network...')
network = models.Mnist(num_classes=10, input_dim=784)

print("# >>> Move to device...")
network = network.to(args.device)

print('# >>> Try to load pretrained model...')
if os.path.exists(args.pretrained):
    pretrained = torch.load(args.pretrained)
    if 'state_dict' in pretrained:
        state_dict = pretrained['state_dict']
    else:
        state_dict = pretrained
    network.load_state_dict(state_dict)
    print(f"Loaded pretrained model from {args.pretrained}")

print('# >>> Try to load checkpoint...')
# get the latest checkpoint name
ckpt_files = glob(os.path.join(args.ckpt))
if len(ckpt_files) == 0:
    print("No checkpoint found.")
    last_rounds = -1
else:
    versions = [int(n.split('.')[-1]) for n in ckpt_files]
    latest_vertion = max(versions)
    for i, v in enumerate(versions):
        if v == latest_vertion:
            break
    ckpt_file = ckpt_files[i]

    # Load checkpoint
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(ckpt['state_dict'])
    last_rounds = ckpt['last_rounds']
    print(f"Loaded checkpoint from {ckpt_file}")

print('# >>> Federated Optimizer...')
if args.role == follower:
    lr = args.follower_lr
    optim = args.follower_optim
else:
    lr = args.leader_lr
    optim = args.leader_optim

optimizer, aggregator = build_optim(
    args.optim,
    network.parameters(),
    lr=args.leader_lr if openfed.core.is_leader(
        args.role) else args.follower_lr,
    role=args.role)

if args.role == leader:
    print('# >>> AutoReducer...')
    auto_reducer = AutoReducerJson(
        weight_key='instances',
        reduce_keys=['accuracy', 'loss', 'duration', 'duration_acg'],
        ignore_keys=["part_id"],
        log_file=os.path.join(args.log_dir, 'mnist.json'))

    print("# >>> Container...")
    container = openfed.container.build_container(aggregator, auto_reducer)
else:
    container = None

print('# >>> World...')
world = World(
    role=args.role,
    async_op='false',
    dal=False,
    mtt=5,
)
print(world)

print('# >>> API...')
openfed_api = openfed.API(
    world=world,
    state_dict=network.state_dict(keep_vars=True),
    pipe=pipe,
    container=container)

print('# >>> Register step functions...')
with openfed_api:
    print('# >>> Synchronize LR Scheduler across different devices...')
    lr_sch = []
    if fl_lr_sch is not None:
        # A hook to automatically load the latest learning rate from leader.
        openfed.hooks.LRTracker(fl_lr_sch)
        lr_sch.append(fl_lr_sch)
    if ld_lr_sch is not None:
        lr_sch.append(ld_lr_sch)

    # Train samples at each round
    assert args.samples or args.sample_ratio
    test_samples = test_dataset.total_parts if args.test_samples is None else args.test_samples
    samples = args.samples if args.samples is not None else int(
        train_dataset.total_parts * args.sample_ratio)

    # A trigger to alarm aggregate operation
    openfed.hooks.Aggregate(
        count=[samples, test_samples],
        checkpoint=args.ckpt,
        lr_scheduler=lr_sch)

    # Some post process after download.
    openfed.hooks.Download()
    # The core step that arrange simulation process.
    openfed.hooks.Dispatch(
        samples=samples,
        parts_list=train_dataset.total_parts,
        test_samples=test_samples,
        test_parts_list=test_dataset.total_parts)
    # The condition to terminate the training process.
    openfed.hooks.Terminate(max_version=args.rounds)

print('# >>> Address...')
address = openfed.build_address(
    backend=args.fed_backend,
    init_method=args.fed_init_method,
    world_size=args.fed_world_size,
    rank=args.fed_rank,
    group_name=args.fed_group_name,
)
print(address)

print("# >>> Connecting...")
openfed_api.build_connection(address=address)


@openfed.api.device_offline_care
def follower_loop():
    # build a trainer and tester
    trainer = Trainer(openfed_api, network, optimizer, train_dataloader)
    tester = Tester(openfed_api, network, test_dataloader)
    task_info = openfed.TaskInfo()

    while True:
        if not openfed_api.transfer(to=False, task_info=task_info):
            break

        if task_info.train:
            trainer.start_training(task_info)

            duraton_acg = trainer.acg_epoch(max_samples=args.acg_samples)
            acc, loss, duration = trainer.train_epoch(epoch=args.epochs)

            task_info.accuracy = acc
            task_info.loss = loss
            task_info.duration = duration
            task_info.duration_acg = duration_acg
            task_info.version += 1

            trainer.finish_training(task_info)
        else:
            tester.start_testing(task_info)

            acc, loss, duration = tester.test_epoch()

            task_info.accuracy = acc
            task_info.loss = loss
            task_info.duration = duration
            task_info.version += 1

            tester.finish_testing(task_info)


# Context `with openfed_api` will go into the specified settings about openfed_api.
# Otherwise, will use the default one which shared by global OpenFed world.
if openfed_api.leader:
    openfed_api.run()
    print('# >>> Finished.')
    openfed_api.finish()
    # Wait all nodes to exit.
    time.sleep(1.0)
else:
    follower_loop()
