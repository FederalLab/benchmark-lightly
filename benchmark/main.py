# type: ignore
import os
from glob import glob
import sys
sys.path.insert(0, "/Users/densechen/code/OpenFed")
import argparse

import openfed
import torch
import torch.nn.functional as F
from openfed import TaskInfo, time_string
from openfed.core import follower, leader, World
from openfed.data import Analysis
from torch.utils.data import DataLoader

import benchmark.datasets as datasets
from benchmark.reducer import AutoReducerJson
from benchmark.utils import StoreDict

# >>> Get default arguments from OpenFed
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser("OpenFed")

# Add parser to address
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

# >>> dataset related
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

# >>> train related
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
parser.add_argument('--ft_optim',
                    '--frontend_optimizer',
                    type=str,
                    default='sgd',
                    choices=['adam', 'sgd'],
                    help='optimizer used in frontend.')
parser.add_argument('--bk_optim',
                    '--backend_optimizer',
                    type=str,
                    default='sgd',
                    choices=['adam', 'sgd'],
                    help='optimizer used in backend.')
parser.add_argument('--penal',
                    type=str,
                    default='none',
                    choices=['none', 'elastic', 'prox', 'scaffold'],
                    help='The penal used to regularize training gradients.')
parser.add_argument('--agg',
                    '--aggregator',
                    type=str,
                    default='average',
                    choices=['average', 'naive', 'elastic'],
                    help='The aggregator used to collect models.')
parser.add_argument('--ft_lr',
                    '--frontend_learning_rate',
                    type=float,
                    default=1e-3,
                    help='The learning rate of frontend optimizer.')
parser.add_argument('--bk_lr',
                    '--backend_learning_rate',
                    type=float,
                    default=1.0,
                    help='The learning rate of backend optimizer.')
parser.add_argument('--ft_lr_sch',
                    '--frontend_learning_rate_scheduler',
                    type=str,
                    default='none',
                    choices=['none', 'multi_step', 'cosine'],
                    help='The frontend learning rate scheduler. (shared among all clients.)')
parser.add_argument('--bk_lr_sch',
                    '--backend_learning_rate_scheduler',
                    type=str,
                    default='none',
                    choices=['none', 'multi_step', 'cosine'],
                    help='The backend learning rate scheduler. (used for server update.)')
parser.add_argument('--bz',
                    '--batch_size',
                    type=int,
                    default=10,
                    help='The batch size.')
parser.add_argument('--gpu',
                    action='store_true',
                    default=False,
                    help="Whether to use gpu.")

# >>> log related
parser.add_argument('--log_level',
                    type=str,
                    default='SUCCESS',
                    choices=['SUCCESS', 'INFO', 'DEBUG', 'ERROR'],
                    help='The log level of openfed backend.')
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

# >>> Task related
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

# >>> set log level
openfed.logger.log_level(level=args.log_level)
# openfed.logger.log_level(level="DEBUG")
openfed.utils.seed_everything(args.seed)

args.ft = args.fed_rank > 0
print(args)
print("# >>> Config Dataset...")
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

print('# >>> Create DataLoader...')
train_dataloader = DataLoader(
    train_dataset, batch_size=args.bz, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=args.bz, shuffle=False, num_workers=0, drop_last=False)

print('# >>> Specify device...')
if args.gpu and torch.cuda.is_available():
    args.gpu = args.fed_rank % torch.cuda.device_count()
    args.device = torch.device(args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device('cpu')

print(f"Use {args.device}")

print('# >>> Define network for specified tasks...')
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

network = network.to(args.device)
loss_fn = loss_fn.to(args.device)

print('# >>> Load pretrained model...')
if os.path.exists(args.pretrained):
    pretrained = torch.load(args.pretrained)
    if 'state_dict' in pretrained:
        state_dict = pretrained['state_dict']
    else:
        state_dict = pretrained
    network.load_state_dict(state_dict)
    print("Loaded pretrained model.")

print('# >>> Load checkpoint...')
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

    print(f"Loaded checkpoint: {ckpt_file}")

    # Load checkpoint
    ckpt = torch.load(ckpt_file)

    network.load_state_dict(ckpt['state_dict'])

    last_rounds = ckpt['last_rounds']

print('# >>> Optimizer...')
if args.ft:
    if args.ft_optim == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(
        ), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
    elif args.ft_optim == 'adam':
        optimizer = torch.optim.Adam(
            network.parameters(), lr=args.ft_lr, betas=(0.9, 0.999))
    else:
        raise NotImplementedError
else:
    if args.bk_optim == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(
        ), lr=args.bk_lr, momentum=0.9, weight_decay=1e-4)
    elif args.bk_optim == 'adam':
        optimizer = torch.optim.Adam(
            network.parameters(), lr=args.bk_lr, betas=(0.9, 0.999))
    else:
        raise NotImplementedError


print('# >>> Penal...')
if args.agg == 'elastic':
    # elastic aggregator needs elastic penal.
    args.penal = 'elastic'
if args.penal == 'none':
    penalizer = None
elif args.penal == 'elastic':
    penalizer = openfed.pipe.ElasticPenalizer(ft=args.ft, momentum=0.9)
elif args.penal == 'prox':
    penalizer = openfed.pipe.ProxPenalizer(ft=args.ft, mu=0.9)
elif args.penal == 'scaffold':
    penalizer = openfed.pipe.ScaffoldPenalizer(ft=args.ft, lr=args.ft_lr)
else:
    raise NotImplementedError

print('# >>> Build Pipe...')
pipe = openfed.pipe.build_pipe(optimizer, penalizer)

if args.penal == 'scaffold':
    pipe.init_c_para()

print('# >>> Scheduler...')
if args.ft_lr_sch == 'none':
    ft_lr_sch = None
elif args.ft_lr_sch == 'cosine':
    ft_lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.rounds, last_epoch=last_rounds)
elif args.ft_lr_sch == 'multi_step':
    ft_lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.rounds * 0.5), int(args.rounds * 0.8)], gamma=0.1, last_epoch=last_rounds
    )
else:
    raise NotImplementedError

if args.bk_lr_sch == 'none':
    bk_lr_sch = None
elif args.bk_lr_sch == 'cosine':
    bk_lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.rounds, last_epoch=last_rounds)
elif args.bk_lr_sch == 'multi_step':
    bk_lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.rounds * 0.5), int(args.rounds * 0.8)], gamma=0.1, last_epoch=last_rounds
    )
else:
    raise NotImplementedError

print('# >>> Aggregator...')
if not args.ft:
    # compute other keys to track
    if args.ft_optim == 'sgd':
        other_keys = ['momentum_buffer']
    elif args.ft_optim == 'adam':
        other_keys = ['exp_avg', 'exp_avg_sq']
    else:
        raise NotImplementedError
    if args.penal == 'scaffold':
        other_keys.append('c_para')
    if args.agg == 'average':
        aggregator = openfed.container.AverageAgg(
            network.parameters(), other_keys=other_keys)
    elif args.agg == 'naive':
        aggregator = openfed.container.NaiveAgg(
            network.parameters(), other_keys=other_keys)
    elif args.agg == 'elastic':
        aggregator = openfed.container.ElasticAgg(
            network.parameters(), other_keys=other_keys)
    else:
        raise NotImplementedError

    print('# >>> AutoReducer...')
    auto_reducer = AutoReducerJson(
        weight_key='instances',
        reduce_keys=['accuracy', 'loss'],
        ignore_keys=["part_id"],
        log_file=args.log_file)

    print("# >>> Build Container...")
    container = openfed.container.build_container(aggregator, auto_reducer)
else:
    container = None

print('# >>> Specify an API for building federated learning...')
openfed_api = openfed.API(
    world=World(role=follower if args.ft else leader),
    state_dict=network.state_dict(keep_vars=True),
    pipe=pipe,
    container=container)

print('# >>> Register step functions...')
with openfed_api:
    print('# >>> Synchronize LR Scheduler across different Frontends...')
    lr_sch = []
    if ft_lr_sch is not None:
        openfed.hooks.LRTracker(ft_lr_sch)
        lr_sch.append(ft_lr_sch)
    if bk_lr_sch is not None:
        lr_sch.append(bk_lr_sch)

    # Train samples at each round
    assert args.samples or args.sample_ratio
    test_samples = test_dataset.total_parts if args.test_samples is None else args.test_samples
    samples = args.samples if args.samples is not None else int(
        train_dataset.total_parts * args.sample_ratio)

    openfed.hooks.Aggregate(
        count=[samples, test_samples],
        checkpoint=args.ckpt,
        lr_scheduler=lr_sch)

    openfed.hooks.Download()
    openfed.hooks.Dispatch(
        samples=samples,
        parts_list=train_dataset.total_parts,
        test_samples=test_samples,
        test_parts_list=test_dataset.total_parts)

    openfed.hooks.Terminate(max_version=args.rounds)

print('# >>> Connect to Address...')
openfed_api.build_connection(
    address=openfed.build_address(
        backend     = args.fed_backend,
        init_method = args.fed_init_method,
        world_size  = args.fed_world_size,
        rank        = args.fed_rank,
        group_name  = args.fed_group_name,
    ))


print('# >>> Train Dataset')
Analysis.digest(train_dataset)

print('# >>> Test Dataset')
Analysis.digest(test_dataset)


def frontend_loop():
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
            if args.agg == 'elastic':
                assert pipe, "pipe must be specified."
                network.train()

                for data in train_dataloader:
                    input, target = data
                    input, target = input.to(
                        args.device), target.to(args.device)

                    pipe.zero_grad()
                    output = network(input)
                    F.mse_loss(output, torch.zeros_like(output)).backward()
                    pipe.acg()

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
        if pipe is not None and task_info.train:
            pipe.clear_buffer()


# Context `with openfed_api` will go into the specified settings about openfed_api.
# Otherwise, will use the default one which shared by global OpenFed world.
with openfed_api:
    if openfed_api.leader:
        openfed_api.run()
    else:
        frontend_loop()

print('# >>> Finished.')
openfed_api.finish()