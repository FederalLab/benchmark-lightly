# type: ignore
import os
from glob import glob
import sys
sys.path.insert(0, "/Users/densechen/code/OpenFed")
import argparse
import time
import openfed
import torch
import torch.nn.functional as F
from openfed import TaskInfo, time_string
from openfed.core import follower, leader, World
from openfed.data import Analysis
from torch.utils.data import DataLoader
from pprint import pprint
import benchmark.datasets as datasets
from benchmark.reducer import AutoReducerJson
from benchmark.utils import StoreDict

parser = argparse.ArgumentParser("OpenFed")

# address
parser.add_argument(
    "--fed_backend",
    default="gloo",
    type=str,
    choices=["gloo", "mpi", "nccl"], )
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
parser.add_argument("--dataset",
                    type=str,
                    default='mnist',
                    choices=['cifar100', 'emnist', 'mnist',
                             'shakespeare', 'stackoverflow'],
                    help='The dataset to train. `cifar100` is copied from TTF,'
                    '`mnist` can be used with different partition methods,'
                    '`emnist` is a federated dataset for image classification,'
                         '`shakespear` and `stackoverflow` are the federated datasets about nlp.'
                    )
parser.add_argument('--data_root',
                    type=str,
                    default="data",
                    help='The folder contains all datasets.')
parser.add_argument('--download',
                    action='store_true',
                    default=False,
                    help='Whether to download dataset if it is not existing.')
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
parser.add_argument('--follower_optim',
                    type=str,
                    default='sgd',
                    choices=['adam', 'sgd'],
                    help='optimizer used in follower.')
parser.add_argument('--leader_optim',
                    type=str,
                    default='sgd',
                    choices=['adam', 'sgd'],
                    help='optimizer used in leader.')
parser.add_argument('--penal',
                    type=str,
                    default='none',
                    choices=['none', 'prox', 'scaffold'],
                    help='The penal used to regularize training gradients.')
parser.add_argument('--agg',
                    '--aggregator',
                    type=str,
                    default='average',
                    choices=['average', 'naive', 'elastic'],
                    help='The aggregator used to collect models.')
parser.add_argument('--follower_lr',
                    type=float,
                    default=1e-3,
                    help='The learning rate of follower optimizer.')
parser.add_argument('--leader_lr',
                    type=float,
                    default=1.0,
                    help='The learning rate of leader optimizer.')
parser.add_argument('--follower_lr_sch',
                    type=str,
                    default='none',
                    choices=['none', 'multi_step', 'cosine'],
                    help='The follower learning rate scheduler. (shared with all clients.)')
parser.add_argument('--leader_lr_sch',
                    type=str,
                    default='none',
                    choices=['none', 'multi_step', 'cosine'],
                    help='The leader learning rate scheduler. (used for server update.)')
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
parser.add_argument('--log_file',
                    type=str,
                    default='logs/benchmark.json',
                    help="The file to log train and test information.")
parser.add_argument('--pretrained',
                    type=str,
                    default='',
                    help='The path to pretrained model.')
parser.add_argument('--ckpt',
                    type=str,
                    default='/tmp/openfed',
                    help='The folder to save checkpoints.')

# task
parser.add_argument('--task',
                    type=str,
                    default='cls',
                    choices=['cls', 'nwp', 'ncp', 'ae', 'tp'],
                    help='Task to perform experiments.'
                    'cls: Image Classification. '
                    'nwp: Next Word Prediction. '
                         'ncp: Next Word Prediction. '
                         'ae: AutoEncoder. '
                         'tp: Tag Prediction. '
                    )
parser.add_argument('--network',
                    type=str,
                    default='lr',
                    choices=['emnist', 'emnist_dp', 'emnist_lr',
                             'emnist_ae', 'lr', 'shakespeare_ncp', 'stackoverflow_nwp'],
                    help='Which network used to train.')
parser.add_argument('--network_cfg',
                    nargs='+',
                    type=str,
                    default='',
                    action=StoreDict,
                    help='''
                    In : args1: 0.0 args2: "dict(a=1)"
                    Out: {'args1': 0.0, arg2: dict(a=1)}
                    ''')
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

print("# >>> Dataset...")
if args.dataset == "cifar100":
    train_dataset = datasets.get_cifar100(root=args.data_root, train=True)
    test_dataset = datasets.get_cifar100(root=args.data_root, train=False)
elif args.dataset == 'emnist':
    train_dataset = datasets.get_emnist(root=args.data_root, train=True)
    test_dataset = datasets.get_emnist(root=args.data_root, train=False)
elif args.dataset == 'mnist':
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
elif args.dataset == 'shakespeare':
    if args.task == 'ncp':
        train_dataset = datasets.get_shakespeare_ncp(
            root=args.root, train=True)
        test_dataset = datasets.get_shakespeare_ncp(
            root=args.root, train=False)
    elif args.task == 'nwp':
        train_dataset = datasets.get_shakespeare_nwp(
            root=args.data_root, train=True)
        test_dataset = datasets.get_shakespeare_nwp(
            root=args.data_root, train=False)
    else:
        raise NotImplementedError
elif args.dataset == 'stackoverflow':
    if args.task == 'nwp':
        train_dataset = datasets.get_stackoverflow_nwp(
            root=args.data_root, train=True)
        test_dataset = datasets.get_stackoverflow_nwp(
            root=args.data_root, train=False)
    elif args.task == 'tp':
        train_dataset = datasets.get_stackoverflow_tp(
            root=args.data_root, train=True)
        test_dataset = datasets.get_stackoverflow_tp(
            root=args.data_root, train=False)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError

print('# >>> Train Dataset:')
Analysis.digest(train_dataset)

print('# >>> Test Dataset:')
Analysis.digest(test_dataset)

print('# >>> DataLoader...')
train_dataloader = DataLoader(
    train_dataset, batch_size=args.bz, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=args.bz, shuffle=False, num_workers=0, drop_last=False)

print('# >>> Device...')
if args.gpu and torch.cuda.is_available():
    args.gpu = args.fed_rank % torch.cuda.device_count()
    args.device = torch.device(args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device('cpu')

print(f"Let's use {args.device}.")

print('# >>> Network...')
if args.network == 'lr':
    from benchmark.models.lr import LogisticRegression, acc_fn, loss_fn
    network = LogisticRegression(**args.network_cfg)
elif args.network == 'emnist':
    from benchmark.models.emnist import EMNIST, acc_fn, loss_fn
    network = EMNIST(**args.network_cfg)
elif args.network == 'emnist_dp':
    from benchmark.models.emnist import EMNISTDropout, acc_fn, loss_fn
    network = EMNISTDropout(**args.network_cfg)
elif args.network == 'emnist_lr':
    from benchmark.models.emnist import EMNISTLinear, acc_fn, loss_fn
    network = EMNISTLinear(**args.network_cfg)
elif args.network == 'emnist_ae':
    from benchmark.models.emnist_ae import EMNISTAE, acc_fn, loss_fn
    network = EMNISTAE(**args.network_cfg)
elif args.network == 'shakespeare_ncp':
    from benchmark.models.shakespeare import ShakespeareNCP, acc_fn, loss_fn
    network = ShakespeareNCP(**args.network_cfg)
elif args.network == 'stackoverflow_nwp':
    from benchmark.models.stackoverflow import (StackOverFlowNWP, acc_fn,
                                                loss_fn)
    network = StackOverFlowNWP(**args.network_cfg)
else:
    raise NotImplementedError

print("# >>> Move to device...")
network = network.to(args.device)
loss_fn = loss_fn.to(args.device)

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

print('# >>> Optimizer...')
if args.role == follower:
    lr = args.follower_lr
    optim = args.follower_optim
else:
    lr = args.leader_lr
    optim = args.leader_optim

if optim == 'sgd':
    optimizer = torch.optim.SGD(network.parameters(
    ), lr=lr, momentum=0.9, weight_decay=1e-4)
elif optim == 'adam':
    optimizer = torch.optim.Adam(
        network.parameters(), lr=lr, betas=(0.9, 0.999))
else:
    raise NotImplementedError

# compute other keys to track
if optim == 'sgd':
    other_keys = ['momentum_buffer']
elif optim == 'adam':
    other_keys = ['exp_avg', 'exp_avg_sq']
else:
    raise NotImplementedError

if args.penal == 'scaffold':
    other_keys.append('c_para')

print('# >>> Penalizer...')
if args.penal == 'none':
    penalizer = openfed.pipe.Penalizer(
        role=args.role, 
        pack_set=other_keys,
        unpack_set=other_keys,
    )
elif args.penal == 'prox':
    penalizer = openfed.pipe.ProxPenalizer(
        role=args.role, 
        mu=0.9, 
        pack_set=other_keys, 
        unpack_set=other_keys)
elif args.penal == 'scaffold':
    penalizer = openfed.pipe.ScaffoldPenalizer(
        role=args.role, 
        lr=args.follower_lr,
        pack_set=other_keys,
        unpack_set=other_keys)
else:
    raise NotImplementedError

if args.agg == 'elastic':
    # elastic aggregator needs elastic penal.
    elastic_penalizer = openfed.pipe.ElasticPenalizer(role=args.role, momentum=0.9)
    if penalizer is not None:
        penalizer = openfed.glue(elastic_penalizer, penalizer, extra_func=dict(acg_step=None))
    else:
        penalizer = elastic_penalizer

print('# >>> Pipe...')
pipe = openfed.pipe.build_pipe(optimizer, penalizer)

if args.penal == 'scaffold':
    pipe.init_c_para()

def build_lr_sch(lr_sch):
    if lr_sch == 'none':
        lr_sch = None
    elif lr_sch == 'cosine':
        lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max      = args.rounds,
            last_epoch = last_rounds
        )
    elif lr_sch == 'multi_step':
        lr_sch = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones = [int(args.rounds * 0.5), int(args.rounds * 0.8)],
            gamma      = 0.1,
            last_epoch = last_rounds
        )
    else:
        raise NotImplementedError
    return lr_sch

print('# >>> Scheduler...')
fl_lr_sch = build_lr_sch(args.follower_lr_sch)
ld_lr_sch = build_lr_sch(args.leader_lr_sch)

if args.role == leader:
    print('# >>> Aggregator...')
    if args.agg == 'average':
        aggregator = openfed.container.AverageAgg(
            network.parameters(), 
            other_keys=other_keys)
    elif args.agg == 'naive':
        aggregator = openfed.container.NaiveAgg(
            network.parameters(), 
            other_keys=other_keys)
    elif args.agg == 'elastic':
        aggregator = openfed.container.ElasticAgg(
            network.parameters(), 
            other_keys=other_keys)
    else:
        raise NotImplementedError

    print('# >>> AutoReducer...')
    auto_reducer = AutoReducerJson(
        weight_key='instances',
        reduce_keys=['accuracy', 'loss'],
        ignore_keys=["part_id"],
        log_file=args.log_file)

    print("# >>> Container...")
    container = openfed.container.build_container(aggregator, auto_reducer)
else:
    container = None

print('# >>> World...')
world = World(
    role     = args.role,
    async_op = 'false',
    dal      = False,
    mtt      = 5,
)
print(world)

print('# >>> API...')
openfed_api = openfed.API(
    world      = world,
    state_dict = network.state_dict(keep_vars=True),
    pipe       = pipe,
    container  = container)

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
    backend     = args.fed_backend,
    init_method = args.fed_init_method,
    world_size  = args.fed_world_size,
    rank        = args.fed_rank,
    group_name  = args.fed_group_name,
)
print(address)

print("# >>> Connecting...")
openfed_api.build_connection(address=address)

@openfed.api.device_offline_care
def follower_loop():
    while True:
        task_info = TaskInfo()
        # Download latest model.
        print(f"{time_string()}: Downloading latest model from server.")
        if not openfed_api.transfer(to=False, task_info=task_info):
            print(f"Downloading failed.")
            break

        # Downloaded
        print(f"{time_string()}: Downloaded!")
        print(task_info)
        
        part_id, version = task_info.part_id, task_info.version

        total_loss = []
        correct = []
        if task_info.train:
            train_dataset.set_part_id(part_id)
            task_info.instances = len(train_dataset)
            # Compute necessary infomation about dataset for federated learning.
            # if args.agg == 'elastic':
            #     assert pipe, "pipe must be specified."
            #     network.train()

            #     for data in train_dataloader:
            #         input, target = data
            #         input, target = input.to(
            #             args.device), target.to(args.device)

            #         pipe.zero_grad()
            #         output = network(input)
            #         F.mse_loss(output, torch.zeros_like(output)).backward()
            #         pipe.acg()

            # if args.penal == 'scaffold':
            #     # accumulate gradient
            #     network.train()

            #     for data in train_dataloader:
            #         input, target = data
            #         input, target = input.to(
            #             args.device), target.to(args.device)
            #         pipe.zero_grad()
            #         loss_fn(network(input), target).backward()
            #         pipe.acg()

            # Train
            network.train()
            for data in train_dataloader:
                input, target = data
                input, target = input.to(args.device), target.to(args.device)

                # Start a standard forward/backward pass.
                pipe.zero_grad()
                output = network(input)
                loss = loss_fn(output, target)

                loss.backward()

                pipe.step()
                correct.append(acc_fn(target, output))

                total_loss.append(loss.item())
            # Round
            if pipe is not None:
                pipe.round()
        else:
            test_dataset.set_part_id(part_id)
            task_info.instances = len(test_dataset)
            # Test
            with torch.no_grad():
                network.eval()
                for data in test_dataloader:
                    input, target = data
                    input, target = input.to(
                        args.device), target.to(args.device)

                    # Start a standard forward inference pass.
                    output = network(input)
                    loss = loss_fn(output, target)
                    correct.append(acc_fn(target, output))
                    total_loss.append(loss.item())

        total_loss = sum(total_loss)/len(total_loss)
        accuracy = sum(correct) / len(correct)

        task_info.loss = total_loss
        task_info.accuracy = accuracy
        task_info.version = version + 1

        # Set task info
        openfed_api.update_version(version + 1)

        # Upload trained model
        print(f"{time_string()}: Uploading trained model to server.")
        if not openfed_api.transfer(to=True, task_info=task_info):
            print("Uploading failed.")
            break
        print(f"{time_string()}: Uploaded!")

        # Clear state
        if task_info.train:
            pipe.clear_buffer()


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