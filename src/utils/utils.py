import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


def wandb_login(key: Optional[str] = None) -> bool:
    """
    Log in to Weights & Biases (wandb) using an API key.

    Args:
        key (Optional[str]): The API key for logging in. If None, uses the existing login credentials.

    Returns:
        bool: True if login was successful, False otherwise.
    """
    return wandb.login(key=key)


def init_wandb(models, cfg: DictConfig):
    import pytz
    
    try:
        if not cfg.logger.get("project"):
            cfg.logger.project = 'default_project'
        
        if not cfg.logger.get("run_name"):
            now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")
            lr = cfg.optimizer.lr
            cfg.model_save_path += f'/{now}'
            cfg.logger.run_name = f"{cfg.model_name}_lr{lr}_{now}"
        
        wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.run_name,
        )
        
        if cfg.logger.get("watch_model", False):
            wandb.watch(
                models=models,
                log="all" if cfg.logger.log_gradients else "parameters",
                log_freq=cfg.logger.log_freq,
            )
    except Exception as e:
        print(f"[Warning] Failed to initialize wandb: {e}")
        
        
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)