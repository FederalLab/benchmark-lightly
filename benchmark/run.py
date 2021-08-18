import os
import sys

sys.path.insert(0, "/Users/densechen/code/OpenFed")
sys.path.insert(0, "/Users/densechen/code/benchmark")

import argparse
import json
import time
from pprint import pprint

import openfed
import torch
from openfed.core import World, follower, is_leader, leader
from openfed.optim import AutoReduceOp
from openfed.tools import build_optim, builder
from torch.utils.data import DataLoader

from benchmark.datasets import build_dataset
from benchmark.models import build_model
from benchmark.tasks import Tester, Trainer
from benchmark.utils import StoreDict

parser = argparse.ArgumentParser("benchmark-lightly")

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

# task
parser.add_argument('--task',
                    type=str,
                    default="mnist",
                    choices=[
                        'celeba', 'cifar100',
                        'femnist', 'mnist',
                        'reddit', 'sent140',
                        'shakespeare', 'stackoverflow',
                        'synthetic'])
parser.add_argument('--network_args',
                    nargs='+',
                    action=StoreDict,
                    default=dict(),
                    help="extra network args passed in.")

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
parser.add_argument('--partition_args', 
                    nargs='+',
                    action=StoreDict,
                    default=dict(),
                    help="extra partition args passed in.")
parser.add_argument('--num_parts',
                    type=int,
                    default=100,
                    help='The number of the parts to split into.')
parser.add_argument('--tst_num_parts',
                    type=int,
                    default=-1,
                    help='The number of the parts to split into.')
parser.add_argument('--dataset_args',
                    nargs='+',
                    action=StoreDict,
                    default=dict(),
                    help="extra dataset args passed in.")

# train
parser.add_argument('--epochs',
                    type=int,
                    default=1,
                    help='The epochs trained on local client.')
parser.add_argument('--rounds',
                    type=int,
                    default=10,
                    help='The total rounds for federated training.')
parser.add_argument('--act_clts',
                    '--activated_clients',
                    type=int,
                    default=10,
                    help='The number of parts used to train at each round.')
parser.add_argument('--act_clts_rat',
                    '--activated_clients_ratio',
                    type=float,
                    default=1.0,
                    help="The portion of parts used to train at each time, in [0, 1]."
                    )
parser.add_argument('--tst_act_clts',
                    '--test_activated_clients',
                    type=int,
                    default=10,
                    help="The number of parts used to test at each round."
                        "If not specified, use full test dataset.")
parser.add_argument('--tst_act_clts_rat',
                    '--test_activated_clients_ratio',
                    type=float,
                    default=1.0,
                    help="The portion of parts used to train at each time, in [0, 1]."
                    )
parser.add_argument('--max_acg_step',
                    type=int,
                    default=-1,
                    help="The number of samples used to compute acg. -1 used all train data.")
parser.add_argument('--optim',
                    type=str,
                    default='fedsgd',
                    choices=list(builder.keys()),
                    help='Specify fed optimizer.')
parser.add_argument('--optim_args',
                    nargs='+',
                    action=StoreDict,
                    default=dict(),
                    help="extra optim args passed in."
                    )
parser.add_argument('--fl_lr',
                    '--follower_lr',
                    type=float,
                    default=1e-2,
                    help='The learning rate of follower optimizer.')
parser.add_argument('--ld_lr',
                    '--leader_lr',
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
                    default=f'logs/',
                    help="The dir to log train and test information.")
parser.add_argument('--exp_name', 
                    type=str,
                    default='default',
                    help='The experiment name.')
parser.add_argument('--seed',
                    type=int,
                    default=0,
                    help="Seed for everything.")

args = parser.parse_args()

args.tst_num_parts = args.tst_num_parts if args.tst_num_parts > 0 else args.fed_world_size-1

print('>>> Set log level...')
openfed.logger.log_level(level=args.log_level)
openfed.utils.seed_everything(args.seed)

args.role = leader if args.fed_rank == 0 else follower

args.log_dir = os.path.join(args.log_dir, args.task, args.exp_name)

os.makedirs(args.log_dir, exist_ok=True)

if is_leader(args.role):
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)

print('>>> Device...')
if args.gpu and torch.cuda.is_available():
    args.gpu = args.fed_rank % torch.cuda.device_count()
    args.device = torch.device(args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device('cpu')

pprint(args.__dict__)

print(f"Let's use {args.device}.")

print(">>> Dataset...")

if args.task == 'mnist':
    if args.partition == 'iid':
        partitioner = openfed.data.IIDPartitioner()
    elif args.partition == 'dirichlet':
        partitioner = openfed.data.DirichletPartitioner(**args.partition_args)
    elif args.partition == 'power-law':
        partitioner = openfed.data.PowerLawPartitioner(**args.partition_args)
    else:
        raise NotImplementedError
    train_args = dict(
        total_parts=args.num_parts,
        partitioner=partitioner
    )
    test_args = dict(
        total_parts=args.tst_num_parts,
        partitioner=partitioner,
    )
elif args.task == 'reddit':
    train_args = dict(mode='train')
    test_args = dict(mode='test')
else:
    train_args = dict(train=True)
    test_args = dict(train=False)

train_dataset = build_dataset(
    args.task, root=args.data_root, **train_args, **args.dataset_args)
test_dataset = build_dataset(
    args.task, root=args.data_root, **test_args, **args.dataset_args)

print('>>> DataLoader...')
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.bz,
    shuffle=True,
    num_workers=0,
    drop_last=False)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.bz,
    shuffle=False,
    num_workers=0,
    drop_last=False)

print('>>> Network...')
network = build_model(args.task, **args.network_args)
pprint(network)

print(">>> Move to device...")
network = network.to(args.device)

print('>>> AutoReducer...')
auto_reducer = AutoReduceOp(
    weight_key='instances',
    reduce_keys=['accuracy', 'loss', 'duration', 'duration_acg'],
    ignore_keys=["part_id"],
    log_file=os.path.join(args.log_dir, f'{args.task}.json'))

print('>>> Federated Optimizer...')
optimizer, aggregator = build_optim(
    args.optim,
    network.parameters(),
    lr=args.ld_lr if is_leader(args.role) else args.fl_lr,
    role=args.role, 
    reducer=auto_reducer,
    **args.optim_args if is_leader(args.role) else dict())

print(">>> Lr Scheduler...")
lr_scheduler = \
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.rounds)

print('>>> World...')
world = World(
    role=args.role,
    dal=True,
    mtt=5,
)
print(world)

print('>>> API...')
openfed_api = openfed.API(
    world=world,
    state_dict=network.state_dict(keep_vars=True),
    fed_optim=optimizer,
    aggregator=aggregator)

print('>>> Register step functions...')
with openfed_api: 
    parts_list=list(range(train_dataset.total_parts))
    act_clts = args.act_clts if args.act_clts > 0 else\
        int(len(parts_list) * args.act_clts_rat)

    tst_parts_list = list(range(test_dataset.total_parts))
    tst_act_clts = args.tst_act_clts if args.tst_act_clts > 0 else\
        int(len(tst_parts_list) * args.tst_act_clts_rat)

    print(f"Train Part: {len(parts_list)}")
    print(f"Activated Train Part: {act_clts}")
    print(f"Test Part: {len(tst_parts_list)}")
    print(f"Activated Test Part: {tst_act_clts}")
    
    openfed.hooks.Dispatch(
        activated_parts=dict(
            train=act_clts,
            test=tst_act_clts,
        ),
        parts_list=dict(
            train=parts_list, 
            test=tst_parts_list),
        max_version=args.rounds,
        clip_grad_norm=1.0,
    )


print('>>> Address...')
address = openfed.build_address(
    backend=args.fed_backend,
    init_method=args.fed_init_method,
    world_size=args.fed_world_size,
    rank=args.fed_rank,
    group_name=args.fed_group_name,
)
pprint(address)

print(">>> Connecting...")
openfed_api.build_connection(address=address)


@openfed.api.device_offline_care
def follower_loop():
    # build a trainer and tester
    trainer = Trainer(
        openfed_api, network, optimizer, train_dataloader, 
        cache_folder=f'/tmp/{args.task}/{args.exp_name}')
    tester = Tester(openfed_api, network, test_dataloader)
    task_info = openfed.TaskInfo()

    while True:
        if not openfed_api.transfer(to=False, task_info=task_info):
            break

        if task_info.mode == 'train': # type: ignore
            trainer.start_training(task_info)

            duration_acg = trainer.acg_epoch(max_acg_step=args.max_acg_step)
            acc, loss, duration = trainer.train_epoch(epoch=args.epochs)

            train_info = dict(
                accuracy     = acc,
                loss         = loss,
                duration     = duration,
                duration_acg = duration_acg,
                version      = task_info.version + 1,           # type: ignore
                instances    = len(trainer.dataloader.dataset), # type: ignore
            )

            task_info.update(train_info)

            trainer.finish_training(task_info)
            lr_scheduler.step(task_info.version) # type: ignore
        else:
            tester.start_testing(task_info)

            acc, loss, duration = tester.test_epoch()

            test_info = dict(
                accuracy  = acc,
                loss      = loss,
                duration  = duration,
                version   = task_info.version + 1,          # type: ignore
                instances = len(tester.dataloader.dataset), # type: ignore
            )
            task_info.update(test_info)

            tester.finish_testing(task_info)


# Context `with openfed_api` will go into the specified settings about openfed_api.
# Otherwise, will use the default one which shared by global OpenFed world.
if openfed_api.leader:
    openfed_api.run()
    print('>>> Finished.')
    openfed_api.finish()
    # Wait all nodes to exit.
    time.sleep(1.0)
else:
    follower_loop()
