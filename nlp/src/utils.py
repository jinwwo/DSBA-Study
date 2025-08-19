import os
import random
from datetime import datetime
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig


def set_wandb(model: nn.Module, cfg: DictConfig):
    if cfg.use_wandb:
        wandb_login(key=cfg.wandb_key)
        init_wandb(model, cfg.train)

    return cfg.use_wandb


def wandb_login(key: Optional[str] = None) -> bool:
    """
    Log in to Weights & Biases (wandb) using an API key.

    Args:
        key (Optional[str]): The API key for logging in. If None, uses the existing login credentials.

    Returns:
        bool: True if login was successful, False otherwise.
    """
    return wandb.login(key=key)


def init_wandb(models, cfg: DictConfig) -> None:
    import pytz
    
    try:
        if not cfg.get("project_name"):
            cfg.project_name = 'default_project'
        
        if not cfg.get("run_name"):
            now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")
            lr = cfg.lr
            cfg.model_save_path += f'/{now}'
            cfg.run_name = f"{cfg.model_name}_lr{lr}_{now}"
        
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
        )
        
        if cfg.get("watch_model", False):
            wandb.watch(
                models=models,
                log="all" if cfg.log_gradients else "parameters",
                log_freq=cfg.log_freq,
            )
    except Exception as e:
        print(f"[Warning] Failed to initialize wandb: {e}")
        
        
def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    

def set_device(device: Union[str, int]) -> torch.device:
    """
    Set CUDA visible devices and return a torch.device.

    Parameters
    ----------
    device : Union[str, int]
        GPU index (e.g., '0' or 0). If CUDA is unavailable, CPU is used.

    Returns
    -------
    torch.device
        The resolved device (cuda:<idx> or cpu).
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else 'cpu'
    return device