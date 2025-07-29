import os

import hydra
from omegaconf import DictConfig, OmegaConf
from sympy import true

from src.trainer.trainer import Trainer
from src.model.builder import build_model
from src.utils.utils import init_wandb, wandb_login, seed_everything
from src.dataset.loader import get_data_loaders


@hydra.main(config_path='configs', config_name='default.yaml')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    # data loader
    train_loader, valid_loader, test_loader = get_data_loaders(cfg.data)
    
    # model
    model = build_model(cfg.model)
    
    if cfg.use_wandb:
        wandb_login(key=cfg.wandb_key)
        init_wandb(model, cfg.train)
    
    trainer = Trainer(
        model = model,
        cfg = cfg.train,
        train_loader= train_loader,
        valid_loader = valid_loader,
        test_loader = test_loader
    )
    OmegaConf.save(config=cfg, f=os.path.join(cfg.train.model_save_path, 'configs.yaml'))
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()