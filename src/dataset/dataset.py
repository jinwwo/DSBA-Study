from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """
    A generalized PyTorch Dataset class for handling image data (e.g., CIFAR-10, TinyImageNet)
    with support for automatic normalization and dataset-specific transform configurations.
    """
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        dataset_name: str,
        train: bool = True,
        resize: int = 224,
        seed: int = 42,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the CustomDataset.
        
        Args:
            data (np.ndarray): Image data with shape (N, H, W, C), values in range [0, 255].
            targets (np.ndarray): Label data with shape (N,).
            dataset_name (str): Name of the dataset (e.g., "cifar10", "tinyimagenet").
            train (bool): Whether the dataset is for training (applies augmentations if True). Default is True.
            resize (int): Target size for resized (square) image. Default is 224.
            seed (int): Random seed for reproducibility. Default is 42.
            mean (Optional[np.ndarray]): Per-channel mean for normalization. If None, computed from data.
            std (Optional[np.ndarray]): Per-channel std deviation for normalization. If None, computed from data.
        """
        self.data = data
        self.targets = targets
        self.dataset_name = dataset_name.lower()
        self.train = train
        self.seed = seed
        self.mean = mean
        self.std = std
        
        if self.mean is None or self.std is None:
            self.mean, self.std = self._compute_mean_std()

        self.transform = self._build_transform(resize)

    def _compute_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and standard deviation for each image channel for normalization.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and standard deviation values for each channel.
        """
        data_normalized = self.data / 255.0
        mean = np.mean(data_normalized, axis=(0, 1, 2))
        std = np.std(data_normalized, axis=(0, 1, 2))
        return mean, std
    
    def _build_transform(self, resize: int) -> transforms.Compose:
        """
        Build the transformation pipeline for training or evaluation.

        Args:
            resize (int): Target size (square) for resizing the images.

        Returns:
            torchvision.transforms.Compose: The composed transformation pipeline.
        """
        interpolation = transforms.InterpolationMode.BICUBIC
        
        if self.train:
            if self.dataset_name == "tinyimagenet":
                return transforms.Compose([
                    transforms.Resize((256, 256), interpolation=interpolation),
                    transforms.RandomCrop(resize),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])
            else:  # e.g., CIFAR-10 or similar small-scale datasets
                return transforms.Compose([
                    transforms.Resize((resize, resize), interpolation=interpolation),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])
        else: # test
            return transforms.Compose([
                transforms.Resize((resize, resize), interpolation=interpolation),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and its corresponding label tensor.
        """
        image = Image.fromarray(self.data[idx]).convert("RGB")
        label = int(self.targets[idx])

        return self.transform(image), torch.tensor(label, dtype=torch.long)