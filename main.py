import os
from glob import glob

import openfed
import openfed.api as of_api
import openfed.container as fed_container
import openfed.data as fed_data
import openfed.pipe as fed_pipe
import torch
from openfed.common.task_info import TaskInfo
from openfed.core.inform import LRTracker
from openfed.utils import time_string
from torch.utils.data import DataLoader

from benchmark.datasets import (get_cifar100, get_emnist, get_mnist,
                                get_shakespeare_ncp, get_shakespeare_nwp,
                                get_stackoverflow_nwp, get_stackoverflow_tp)
from benchmark.utils import StoreDict

# >>> set log level
openfed.logger.log_level(level="SUCCESS")

# >>> Get default arguments from OpenFed
parser = openfed.parser

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
parser.add_argument('--pipe',
                    type=str,
                    default='none',
                    choices=['none', 'elastic', 'prox', 'scaffold'],
                    help='The pipe used to regularize training gradients.')
parser.add_argument('--agg',
                    '--aggregator',
                    type=str,
                    default='average',
                    choices=['average', 'naive', 'elastic'],
                    help='The aggregator used to collect models.')
parser.add_argument('--ft_lr',
                    '--frontend_learning_rate',
                    type=float,
                    default=0.1,
                    help='The learning rate of frontend optimizer.')
parser.add_argument('--bk_lr',
                    '--backend_optimizer',
                    type=float,
                    default=0.1,
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
parser.add_argument('--pretrained',
                    type=str,
                    default='',
                    help='The path to pretrained model.')
parser.add_argument('--ckpt',
                    type=str,
                    default='/tmp/openfed',
                    help='The folder to save checkpoints.')
parser.add_argument('--wandb',
                    action='store_true',
                    default=False,
                    help='Whether to enable wandb.')
parser.add_argument('--tb',
                    '--tensorboard',
                    action='store_true',
                    default=False,
                    help='Whether to enable tensorboard.')

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
                    type=str,
                    default='',
                    action=StoreDict,
                    help='''
                    In: args1:0.0 args2:"dict(a=1)"
                    Out: {'args1': 0.0, arg2: dict(a=1)}
                    ''')

args = parser.parse_args()

# >>> Config dataset
if args.dataset == "cifar100":
    train_dataset = get_cifar100(root=args.data_root, train=True)
    test_dataset = get_cifar100(root=args.data_root, train=False)
elif args.dataset == 'emnist':
    train_dataset = get_emnist(root=args.data_root, train=True)
    test_dataset = get_emnist(root=args.data_root, train=False)
elif args.dataset == 'mnist':
    if args.partition == 'iid':
        partitioner = fed_data.IIDPartitioner()
    elif args.partition == 'dirichlet':
        partitioner = fed_data.DirichletPartitioner()
    elif args.partition == 'power-low':
        partitioner = fed_data.PowerLawPartitioner()
    else:
        raise NotImplementedError
    train_dataset = get_mnist(
        root=args.data_root, total_parts=args.num_parts, train=True, partitioner=partitioner)
    test_dataset = get_mnist(
        root=args.data_root, total_parts=args.num_parts, train=False, partitioner=partitioner)
elif args.dataset == 'shakespeare':
    if args.task == 'ncp':
        train_dataset = get_shakespeare_ncp(root=args.root, train=True)
        test_dataset = get_shakespeare_ncp(root=args.root, train=False)
    elif args.task == 'nwp':
        train_dataset = get_shakespeare_nwp(root=args.data_root, train=True)
        test_dataset = get_shakespeare_nwp(root=args.data_root, train=False)
    else:
        raise NotImplementedError
elif args.dataset == 'stackoverflow':
    if args.task == 'nwp':
        train_dataset = get_stackoverflow_nwp(root=args.data_root, train=True)
        test_dataset = get_stackoverflow_nwp(root=args.data_root, train=False)
    elif args.task == 'tp':
        train_dataset = get_stackoverflow_tp(root=args.data_root, train=True)
        test_dataset = get_stackoverflow_tp(root=args.data_root, train=False)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError

# >>> Create DataLoader
train_dataloader = DataLoader(
    train_dataset, batch_size=args.bz, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=args.bz, shuffle=False, num_workers=0, drop_last=False)

# >>> Specify device
if args.gpu:
    args.gpu = args.fed_rank % torch.cuda.device_count()
    args.device = torch.device(args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device('cpu')

# >>> Define network for specified tasks.
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

# >>> Load pretrained model
if os.path.exists(args.pretrained):
    pretrained = torch.load(args.pretrained)
    if 'state_dict' in pretrained:
        state_dict = pretrained['state_dict']
    else:
        state_dict = pretrained
    network.load_state_dict(state_dict)
    print("Loaded pretrained model.")

# >>> Load checkpoint
# get the latest checkpoint name
ckpt_files = glob(os.path.join(args.ckpt))
if len(ckpt_files) == 0:
    print("No checkpoint found.")
    last_rounds = 0
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

# >>> Optimizer
if args.ft_optim == 'sgd':
    ft_optim = torch.optim.SGD(network.parameters(
    ), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
elif args.ft_optim == 'adam':
    ft_optim = torch.optim.Adam(
        network.parameters(), lr=args.ft_lr, betas=(0.9, 0.999))
else:
    raise NotImplementedError

if args.bk_optim == 'sgd':
    bk_optim = torch.optim.SGD(network.parameters(
    ), lr=args.bk_lr, momentum=0.9, weight_decay=1e-4)
elif args.bk_optim == 'adam':
    bk_optim = torch.optim.Adam(
        network.parameters(), lr=args.bk_lr, betas=(0.9, 0.999))
else:
    raise NotImplementedError

# >>> Scheduler
if args.ft_lr_sch == 'none':
    ft_lr_sch = None
elif args.ft_lr_sch == 'cosine':
    ft_lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_optim, T_max=args.rounds, last_epoch=last_rounds)
elif args.ft_lr_sch == 'multi_step':
    ft_lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        ft_optim, milestones=[int(args.rounds * 0.5), int(args.rounds * 0.8)], gamma=0.1, last_epoch=last_rounds
    )
else:
    raise NotImplementedError

if args.bk_lr_sch == 'none':
    bk_lr_sch = None
elif args.bk_lr_sch == 'cosine':
    bk_lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        bk_optim, T_max=args.rounds, last_epoch=last_rounds)
elif args.bk_lr_sch == 'multi_step':
    bk_lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        bk_optim, milestones=[int(args.rounds * 0.5), int(args.rounds * 0.8)], gamma=0.1, last_epoch=last_rounds
    )
else:
    raise NotImplementedError

# >>> Aggregator

# compute other keys to track
if args.ft_optim == 'sgd':
    other_keys = ['momentum_buffer']
elif args.ft_optim == 'adam':
    other_keys = ['exp_avg', 'exp_avg_sq']
else:
    raise NotImplementedError

if args.agg == 'average':
    aggregator = fed_container.AverageAgg(
        network.state_dict(keep_vars=True), other_keys=other_keys)
elif args.agg == 'naive':
    aggregator = fed_container.NaiveAgg(
        network.state_dict(keep_vars=True), other_keys=other_keys)
elif args.agg == 'elastic':
    aggregator = fed_container.ElasticAgg(
        network.state_dict(keep_vars=True), other_keys=other_keys)
else:
    raise NotImplementedError

# Pipe
if args.agg == 'elastic':
    # elastic aggregator needs elastic pipe.
    args.pipe = 'elastic'
if args.pipe == 'none':
    pipe = None
elif args.pipe == 'elastic':
    pipe = fed_pipe.ElasticPipe(network.parameters(), momentum=0.9)
elif args.pipe == 'prox':
    pipe = fed_pipe.ProxPipe(network.parameters(), mu=0.9)
elif args.pipe == 'scaffold':
    pipe = fed_pipe.ScaffoldPipe(network.parameters(), lr=args.ft_lr)
else:
    raise NotImplementedError

# >>> Add an auto-reducer to compute task info
auto_reducer = fed_container.AutoReducer(weight_key='instances')

# >>> Specify an API for building federated learning
openfed_api = openfed.API(
    frontend=args.fed_rank > 0,
    state_dict=network.state_dict(keep_vars=True),
    ft_optimizer=ft_optim,
    aggregator=aggregator,
    bk_optimizer=bk_optim,
    pipe=pipe,
    reducer=auto_reducer)

# >>> Synchronize LR Scheduler between Frontend and Backend
lr_tracker = LRTracker(ft_lr_sch)
openfed_api.add_informer_hook(lr_tracker)

# >>> Register more step functions.
with openfed_api:
    # Train samples at each round
    assert args.samples or args.sample_ratio
    test_samples = test_dataset.total_samples if args.test_samples is None else args.test_samples
    samples = args.samples if args.samples is not None else int(
        train_dataset.total_samples * args.sample_ratio)

    of_api.Aggregate(
        count=[samples, test_samples],
        checkpoint=args.ckpt,
        lr_scheduler=[ft_lr_sch, bk_lr_sch])

    of_api.Download()
    of_api.Dispatch(args.total_parts,
                    samples=samples,
                    total_test_parts=test_samples)

    of_api.Terminate(max_version=args.rounds)

# >>> Connect to Address.
openfed_api.build_connection(address=openfed.Address(args=args))


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

        part_id, version = task_info.part_id, task_info.version

        print(f"{time_string()}")
        print(task_info)

        total_loss = []
        correct = []
        if task_info.train:
            train_dataset.set_part_id(part_id)
            task_info.instances = len(train_dataset)
            # Train
            network.train()
            for data in train_dataloader:
                input, target = data
                input, target = input.to(args.device), target.to(args.device)

                # Start a standard forward/backward pass.
                ft_optim.zero_grad()
                output = network(input)
                loss = loss_fn(output, target)

                loss.backward()
                if pipe is not None:
                    pipe.step()

                ft_optim.step()
                correct.append(acc_fn(target, output))

                total_loss.append(loss.item())

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


# Context `with openfed_api` will go into the specified settings about openfed_api.
# Otherwise, will use the default one which shared by global OpenFed world.
with openfed_api:
    if not openfed_api.backend_loop():
        frontend_loop()

print(f"Finished.\nExit Client @{openfed_api.nick_name}.")

# >>> Finished
openfed_api.finish(auto_exit=True)
