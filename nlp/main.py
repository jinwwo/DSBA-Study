import logging
import os

import hydra
import omegaconf
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import get_dataloader
from src.model import EncoderForClassification
from src.utils import init_wandb, seed_everything, wandb_login

_logger = logging.getLogger("train")


def train_iter(model, inputs, optimizer, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits, loss = model(**inputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    accuracy = calculate_accuracy(logits, inputs['label'])
    wandb.log({'train_loss': loss.item(), 'train_acc': accuracy})
    return loss, accuracy


def valid_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    logits, loss = model(**inputs)
    
    accuracy = calculate_accuracy(logits, inputs['label'])
    return loss, accuracy


def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)


def set_device(device):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else 'cpu'
    return device


def get_model_tokenizer(configs, device):
    model = EncoderForClassification(configs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    return model.to(device), tokenizer


@hydra.main(config_path='configs', config_name='default.yaml')
def main(configs : omegaconf.DictConfig):
    print(OmegaConf.to_yaml(configs))
    seed_everything(configs.seed)
    device = set_device(configs.device)
    
    # model, tokenizer
    model, tokenizer = get_model_tokenizer(configs.model, device)
    
    # data loader
    train_loader = get_dataloader(configs.data, tokenizer, 'train')
    valid_loader = get_dataloader(configs.data, tokenizer, 'valid')
    test_loader = get_dataloader(configs.data, tokenizer, 'test')
    
    wandb_login(key=configs.wandb_key)
    init_wandb(model, configs.train)
    
    model_save_path = configs.train.model_save_path
    os.makedirs(model_save_path, exist_ok=True)
    OmegaConf.save(config=configs, f=os.path.join(model_save_path, 'configs.yaml'))
    
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.lr, weight_decay=configs.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)
    
    # Train & validation for each epoch
    epochs = configs.train.max_epochs
    best_valid_accuracy = -1
    for epoch in range(epochs):
        model.train()
        total_train_loss, total_train_acc = 0.0, 0.0
        
        # training
        for batch in tqdm(train_loader, desc=f'Epoch: {epoch+1}/{epochs}'):
            train_loss, train_accuracy = train_iter(model, batch, optimizer, device=device)

            total_train_loss += train_loss.item()
            total_train_acc += train_accuracy

        _logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_train_loss / len(train_loader)}")
        _logger.info(f"Epoch {epoch+1}/{epochs} - Train Acc: {total_train_acc / len(train_loader)}")
        
        # validation
        total_val_loss, total_val_acc = 0.0, 0.0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Validating Epoch" {epoch+1}'):
                valid_loss, valid_accuracy = valid_iter(model, batch, device)
                total_val_loss += valid_loss.item()
                total_val_acc += valid_accuracy
            
        _logger.info(f"Epoch {epoch+1}/{epochs} - Val Loss: {total_val_loss / len(valid_loader)}")
        _logger.info(f"Epoch {epoch+1}/{epochs} - Val Acc: {total_val_acc / len(valid_loader)}")
        wandb.log({'val_loss': total_val_loss / len(valid_loader), 'val_acc': total_val_acc / len(valid_loader)})
        
        if total_val_acc > best_valid_accuracy:
            best_valid_accuracy = total_val_acc            
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pt'))
            _logger.info(f"Best val Acc: {best_valid_accuracy} updating model...")
            _logger.info(f"Best model updated: Saved at {model_save_path}")
        
        wandb.log({'lr': optimizer.param_groups[0]["lr"]})
        scheduler.step()
        
    # testing
    model.eval()
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_model.pt')))
    
    total_test_loss, total_test_acc= 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f'Testing" {epoch+1}'):
            test_loss, test_accuracy = valid_iter(model, batch, device)
            total_test_loss += test_loss.item()
            total_test_acc += test_accuracy
            
    _logger.info(f"Test Loss: {total_test_loss / len(test_loader)}")
    _logger.info(f"Test Acc: {total_test_acc / len(test_loader)}")
    wandb.log({'test_loss': total_test_loss / len(test_loader), 'test_acc': total_test_acc / len(test_loader)})


if __name__ == "__main__":
    main()