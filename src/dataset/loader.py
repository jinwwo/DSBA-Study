import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import CustomDataset


def get_data_loaders(cfg):
    # cifar10, tiny_imagenet에 따라 로직 다르게
    # 1. train/test data, target load
    # 2. train / vaild split
    # 3. train/valid/test dataset class instantiate (CustomDataset)
    # 4. train/valid/test dataloader return
    
    if cfg.dataset.name == 'cifar10':
        train_data, train_target, test_data, test_target = get_cifar10(cfg.dataset.path)
    
    elif cfg.dataset.name == 'tiny_imagenet':
        pass # to do
    
    # train / valid split
    train_data, valid_data, train_target, valid_target = train_test_split(
        train_data,
        train_target,
        test_size=cfg.split_ratio,
        random_state=cfg.seed,
    )
    
    # create datasets
    train_dataset = CustomDataset(
        data = train_data, 
        targets = train_target,
        dataset_name = cfg.dataset.name
    )
    valid_dataset = CustomDataset(
        valid_data,
        valid_target,
        dataset_name = cfg.dataset.name,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    test_dataset = CustomDataset(
        test_data,
        test_target,
        dataset_name = cfg.dataset.name,
        train = False,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def get_cifar10(base_dir):
    def load_batch(file_path):
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        return batch['data'], batch['labels']
    
    def reshape_images(data):
        # (N, 3, 32, 32) -> (N, 32, 32, 3)
        return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Load training batches
    train_data_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_path = os.path.join(base_dir, f'data_batch_{i}')
        data, labels = load_batch(batch_path)
        train_data_list.append(data)
        train_labels_list.append(labels)
    
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    # Load test batch
    test_path = os.path.join(base_dir, 'test_batch')
    test_data, test_labels = load_batch(test_path)
    test_labels = np.array(test_labels)
    
    # Reshape image data to (N, 32, 32, 3)
    train_data = reshape_images(train_data)
    test_data = reshape_images(test_data)
    
    return train_data, train_labels, test_data, test_labels


def get_tinyimagenet(base_dir):
    pass