from typing import Dict, Union

import timm
import torch
from omegaconf import DictConfig

from .model_zoo import SUPPORTED_MODELS


def build_model(cfg: DictConfig) -> torch.nn.Module:
    model_name = cfg.models.name
    num_classes = cfg.models.num_classes
    pretrained = cfg.pretrained
    
    model = timm.create_model(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    return model