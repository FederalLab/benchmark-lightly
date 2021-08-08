import openfed
import torch.optim as optim


def build_fedsgd(parameters, lr, role, **kwargs):
    """Build fedsgd, return optimizer and aggregator (for leader).
    """
    optimizer = optim.SGD(parameters, lr=lr, **kwargs)
    fed_optimizer = openfed.optim.build_fed_optim(optimizer)
    if openfed.core.is_leader(role):
        aggregator = openfed.container.AverageAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedavg(parameters, lr, role, **kwargs):
    """Build fedavg, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(parameters, lr=lr, **kwargs)
    fed_optimizer = openfed.optim.build_fed_optim(optimizer)
    if openfed.core.is_leader(role):
        aggregator = openfed.container.NaiveAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedela(parameters, lr, role, **kwargs):
    """Build fedela, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(parameters, lr=lr, **kwargs)
    penalizer = openfed.optim.ElasticPenalizer(role)
    fed_optimizer = openfed.optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        aggregator = openfed.container.ElasticAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedprox(parameters, lr, role, **kwargs):
    """Build fedprox, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(parameters, lr=lr, **kwargs)
    penalizer = openfed.optim.ProxPenalizer(role)
    fed_optimizer = openfed.optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        aggregator = openfed.container.NaiveAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator


def build_fedscaffold(parameters, lr, role, **kwargs):
    """Build fedscaffold, return optimizer and aggregator (for leader).
    Args:
        kwargs: other parameters for build optimizer.
    """
    optimizer = optim.SGD(parameters, lr=lr, **kwargs)
    penalizer = openfed.optim.ScaffoldPenalizer(role)
    fed_optimizer = openfed.optim.build_fed_optim(optimizer, penalizer)

    if openfed.core.is_leader(role):
        aggregator = openfed.container.NaiveAgg(parameters)
    else:
        aggregator = None

    return fed_optimizer, aggregator
