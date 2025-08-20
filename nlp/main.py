import json
import logging
import os
import time
from typing import Dict, Tuple

import hydra
import omegaconf
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data import get_dataloader
from src.model import EncoderForClassification
from src.utils import seed_everything, set_device, set_wandb

_logger = logging.getLogger("train")


def train_iter(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accelerator: Accelerator,
    step_counter: int
) -> Tuple[torch.Tensor, float]:
    """
    Single training iteration (forward, backward, step) and logging.

    Parameters
    ----------
    model : nn.Module
        Classification model (returns logits, loss).
    inputs : Dict[str, torch.Tensor]
        Batch inputs containing 'input_ids', 'attention_mask', optional 'token_type_ids', and 'label'.
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates.
    device : torch.device
        Target device.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss tensor (before .item()).
    accuracy : float
        Accuracy for the batch.
    """
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with accelerator.accumulate(model):
        logits, loss = model(**inputs)
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_counter += 1

    accuracy = calculate_accuracy(logits, inputs["label"])
    
    return loss, accuracy


def valid_iter(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Single validation iteration (forward only).

    Parameters
    ----------
    model : nn.Module
        Classification model (returns logits, loss).
    inputs : Dict[str, torch.Tensor]
        Batch inputs.
    device : torch.device
        Target device.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss tensor.
    accuracy : float
        Accuracy for the batch.
    """
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits, loss = model(**inputs)

    accuracy = calculate_accuracy(logits, inputs["label"])
    return loss, accuracy


def calculate_accuracy(logits: torch.Tensor, label: torch.Tensor) -> float:
    """
    Compute accuracy from logits and integer class labels.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (batch_size, num_classes).
    label : torch.Tensor
        Shape (batch_size,), dtype=torch.long.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)


def get_model_tokenizer(
    configs: omegaconf.DictConfig, device: torch.device
) -> Tuple[nn.Module, PreTrainedTokenizerBase]:
    """
    Instantiate model and tokenizer, move model to device.

    Parameters
    ----------
    configs : omegaconf.DictConfig
        Model-related config; must include `model_id`, `dropout_rate`, `num_labels`.
    device : torch.device
        Target device.

    Returns
    -------
    (model, tokenizer) : Tuple[nn.Module, PreTrainedTokenizerBase]
        Initialized model on the given device and matching tokenizer.
    """
    model = EncoderForClassification(configs)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    return model.to(device), tokenizer


@hydra.main(config_path="configs", config_name="default.yaml")
def main(configs: omegaconf.DictConfig) -> None:
    """
    Entry point for training/validation/testing pipeline.
    """
    print(OmegaConf.to_yaml(configs))
    seed_everything(configs.seed)
    device = set_device(configs.device)

    # model & tokenizer
    model, tokenizer = get_model_tokenizer(configs.model, device)

    # data loaders
    train_loader = get_dataloader(configs.data, tokenizer, "train")
    valid_loader = get_dataloader(configs.data, tokenizer, "valid")
    test_loader = get_dataloader(configs.data, tokenizer, "test")

    # set wandb
    use_wandb = set_wandb(model, configs)

    # save configs and prepare model save path
    model_save_path = configs.train.model_save_path
    os.makedirs(model_save_path, exist_ok=True)
    OmegaConf.save(config=configs, f=os.path.join(model_save_path, "configs.yaml"))

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs.train.lr, weight_decay=configs.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1, gamma=0.9
    )

    # gradient accumulation
    step_counter = 0
    accelerator = Accelerator(gradient_accumulation_steps=configs.train.gradient_accumulation_steps)
    model, optimizer, scheduler, train_loader, valid_loader, test_loader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_loader, valid_loader, test_loader
        )
    )

    # train & validate
    history = {}
    epochs = configs.train.max_epochs
    best_valid_accuracy = -1
    train_start = time.perf_counter()
    
    for epoch in range(epochs):
        model.train()
        total_train_loss, total_train_acc = 0.0, 0.0

        # train
        for batch in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{epochs}"):
            train_loss, train_accuracy = train_iter(
                model, batch, optimizer, device, accelerator, step_counter
            )

            if use_wandb:
                wandb.log({"train_loss": train_loss.item(), "train_acc": train_accuracy})
                
            total_train_loss += train_loss.item()
            total_train_acc += train_accuracy

        _logger.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {total_train_loss / len(train_loader)}"
        )
        _logger.info(
            f"Epoch {epoch+1}/{epochs} - Train Acc: {total_train_acc / len(train_loader)}"
        )

        # validate
        history[f"epoch_{epoch}"] = []
        total_val_loss, total_val_acc = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc=f'Validating Epoch" {epoch+1}')):                
                valid_loss, valid_accuracy = valid_iter(model, batch, device)
                total_val_loss += valid_loss.item()
                total_val_acc += valid_accuracy
        
        avg_val_loss = total_val_loss / len(valid_loader)
        avg_val_acc = total_val_acc / len(valid_loader)
        
        elapsed_time = time.perf_counter() - train_start
        
        history[f"epoch_{epoch}"].append({
            "time": elapsed_time,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc
        })
                
        _logger.info(
            f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss}"
        )
        _logger.info(
            f"Epoch {epoch+1}/{epochs} - Val Acc: {avg_val_acc}"
        )
        
        if use_wandb:
            wandb.log(
                {
                    "val_loss": total_val_loss / len(valid_loader),
                    "val_acc": total_val_acc / len(valid_loader),
                }
            )

        if avg_val_acc > best_valid_accuracy:
            best_valid_accuracy = avg_val_acc
            torch.save(
                model.state_dict(), os.path.join(model_save_path, "best_model.pt")
            )
            _logger.info(f"Best val Acc: {best_valid_accuracy} updating model...")
            _logger.info(f"Best model updated: Saved at {model_save_path}")

        if use_wandb:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})
        
        scheduler.step()
        with open(os.path.join(model_save_path, "val_history.json"),"w") as f:
            json.dump(history, f, indent=4)
    
    torch.save(
        model.state_dict(), os.path.join(model_save_path, "last_model.pt")
    )
    
    # test using the best checkpoint
    model.eval()
    model.load_state_dict(torch.load(os.path.join(model_save_path, "best_model.pt")))

    total_test_loss, total_test_acc = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Testing" {epoch+1}'):
            test_loss, test_accuracy = valid_iter(model, batch, device)
            total_test_loss += test_loss.item()
            total_test_acc += test_accuracy

    _logger.info(f"Test Loss: {total_test_loss / len(test_loader)}")
    _logger.info(f"Test Acc: {total_test_acc / len(test_loader)}")
    
    if use_wandb:
        wandb.log(
            {
                "test_loss": total_test_loss / len(test_loader),
                "test_acc": total_test_acc / len(test_loader),
            }
        )


if __name__ == "__main__":
    main()