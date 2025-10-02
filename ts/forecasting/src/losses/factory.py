import torch.nn as nn

def create_criterion(loss_name: str, params: dict = {}):
    if loss_name == 'mse':
        criterion = nn.MSELoss(**params)
    elif loss_name == 'mae':
        criterion = nn.L1Loss(**params)
    elif loss_name == 'BCE':
        criterion = nn.BCEWithLogitsLoss(**params)
    return criterion