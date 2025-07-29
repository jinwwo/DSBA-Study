import torch
from omegaconf import DictConfig


def create_criterion(cfg: DictConfig) -> torch.nn.Module:
    type = cfg.type.lower()
    
    if type == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif type == 'negative_learning':
        return torch.nn.NLLLoss()
    else:
        raise NotImplementedError(f'Unsupported criterion: {type}')


def create_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    opt_cfg = dict(cfg)
    type = opt_cfg.pop('type').lower()

    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
    }

    if type not in optimizers:
        raise NotImplementedError(f'Unsupported optimizer: {type}')

    return optimizers[type](model.parameters(), **opt_cfg)


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    sched_cfg = dict(cfg)
    sched_type = sched_cfg.pop('type').lower()

    schedulers = {
        'step': torch.optim.lr_scheduler.StepLR,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'multistep': torch.optim.lr_scheduler.MultiStepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
    }

    if sched_type not in schedulers:
        raise NotImplementedError(f'Unsupported scheduler: {sched_type}')

    SchedulerClass = schedulers[sched_type]
    
    return SchedulerClass(optimizer, **sched_cfg)