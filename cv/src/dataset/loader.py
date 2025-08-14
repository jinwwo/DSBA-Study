import os
import pickle
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import CustomDataset


def get_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return PyTorch DataLoaders for training, validation, and testing.
    """
    if cfg.dataset.name == 'cifar10':
        train_data, train_target, test_data, test_target = get_cifar10(cfg.dataset.path)
    
    elif cfg.dataset.name == 'tiny_imagenet':
        train_data, train_target, test_data, test_target = get_tinyimagenet(cfg.dataset.path)
    
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


def get_cifar10(base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the CIFAR-10 dataset from disk.

    Args:
        base_dir (str): Path to the directory containing CIFAR-10 batches.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - train_data: (N_train, 32, 32, 3)
            - train_labels: (N_train,)
            - test_data: (N_test, 32, 32, 3)
            - test_labels: (N_test,) int64
    """
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


def get_tinyimagenet(base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TinyImageNet dataset and return data and labels in (N, H, W, C) format,
    consistent with CIFAR-10 loading format.

    Args:
        base_dir (str): Path to the root of 'tiny-imagenet-200' folder.

    Returns:
        train_data (np.ndarray): shape (N_train, 64, 64, 3), dtype=uint8
        train_labels (np.ndarray): shape (N_train,), dtype=int
        test_data (np.ndarray): shape (N_test, 64, 64, 3), dtype=uint8
        test_labels (np.ndarray): shape (N_test,), dtype=int
    """
    # Read class ids
    wnids_path = os.path.join(base_dir, 'wnids.txt')
    with open(wnids_path, 'r') as f:
        class_ids = [line.strip() for line in f.readlines()]
    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(class_ids)}

    # Load training data
    train_images = []
    train_labels = []

    for cls_id in class_ids:
        img_dir = os.path.join(base_dir, 'train', cls_id, 'images')
        for fname in os.listdir(img_dir):
            if fname.endswith('.JPEG'):
                img_path = os.path.join(img_dir, fname)
                img = Image.open(img_path).convert('RGB')
                train_images.append(np.array(img))
                train_labels.append(class_to_idx[cls_id])

    train_data = np.stack(train_images, axis=0)  # (N, 64, 64, 3)
    train_labels = np.array(train_labels, dtype=np.int64)

    # Load validation data
    val_dir = os.path.join(base_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')

    # Map image file name to class
    img_to_class = {}
    with open(val_annotations_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            fname, cls_id = parts[0], parts[1]
            img_to_class[fname] = class_to_idx[cls_id]

    val_images = []
    val_labels = []

    for fname in os.listdir(val_img_dir):
        if fname.endswith('.JPEG') and fname in img_to_class:
            img_path = os.path.join(val_img_dir, fname)
            img = Image.open(img_path).convert('RGB')
            val_images.append(np.array(img))
            val_labels.append(img_to_class[fname])

    test_data = np.stack(val_images, axis=0)   # (N, 64, 64, 3)
    test_labels = np.array(val_labels, dtype=np.int64)

    return train_data, train_labels, test_data, test_labels


def get_cifar10_ds(base_dir) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the CIFAR-10.1 dataset (v6) for out-of-distribution testing.

    Args:
        base_dir (str): Path to the directory containing 'cifar10.1_v6_data.npy' and 'cifar10.1_v6_labels.npy'.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - test_data: (N, 32, 32, 3) float32 or uint8
            - test_labels: (N,) int64
    """
    test_data = np.load(os.path.join(base_dir, 'cifar10.1_v6_data.npy'))
    test_labels = np.load(os.path.join(base_dir, 'cifar10.1_v6_labels.npy'))
    
    return test_data, test_labels