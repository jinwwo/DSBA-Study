from lion_pytorch import Lion
from torch import optim

def create_optimizer(model, opt_name: str, lr: float, params: dict = {}):
    if opt_name == 'adamw':
        model_optim = optim.AdamW(model.parameters(), lr=lr, **params)
    elif opt_name == 'adam':
        model_optim = optim.Adam(model.parameters(), lr=lr, **params)
    elif opt_name == 'sgd':
        model_optim = optim.SGD(model.parameters(), lr=lr, **params)
    elif opt_name == 'lion':
        model_optim = Lion(model.parameters(), lr=lr, **params)
    return model_optim