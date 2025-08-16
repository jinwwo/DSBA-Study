from typing import Any, Dict, List, Literal, Mapping, Tuple

import omegaconf
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


class IMDBDatset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: omegaconf.DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        split: Literal['train', 'valid', 'test']
    ) -> None:
        """
        A PyTorch Dataset wrapper for the IMDB dataset that loads data,
        splits it into train/valid/test sets, and applies tokenization.

        Parameters
        ----------
        configs : omegaconf.DictConfig
            Configuration containing:
                - dataset_name : str   (dataset name for `load_dataset`, e.g., "imdb")
                - val_size     : float (fraction for validation split)
                - test_size    : float (fraction for test split)
                - seed         : int   (random seed for reproducibility)
                - max_length   : int   (max token length for the tokenizer)
        tokenizer : transformers.PreTrainedTokenizerBase
            Tokenizer used to encode the text.
        split : {"train", "valid", "test"}
            Which subset to load.
        
        Notes
        -----
        - Merges the original IMDB `train` and `test` sets, then splits into
          train/valid and train/test in sequence.
        """
        self.tokenizer = tokenizer
        self.configs = configs

        dataset = load_dataset(configs.dataset_name)
        full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
        
        train_valid_split = full_dataset.train_test_split(configs.val_size, seed=configs.seed)
        train_valid = train_valid_split['train']
        val_data = train_valid_split['test'] # validation set
        
        train_test_split = train_valid.train_test_split(configs.test_size, seed=configs.seed)
        train_data = train_test_split['train'] # training set
        test_data = train_test_split['test'] # testing set
        
        self.dataset = DatasetDict({
            'train': train_data,
            'valid': val_data,
            'test': test_data
        })[split]
        
        self.dataset = self.dataset.map(
            self.tokenize_function, batched=True, remove_columns=['text']
        ).to_dict()
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset['input_ids'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single tokenized sample.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
                - input_ids       : LongTensor [max_length]
                - token_type_ids  : LongTensor [max_length] (if present)
                - attention_mask  : LongTensor [max_length]
                - label           : LongTensor []
        """
        inputs = {key: torch.tensor(self.dataset[key][idx], dtype=torch.long) for key in self.dataset}
        
        return inputs

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader to batch samples together.

        Parameters
        ----------
        batch : List[Dict[str, torch.Tensor]]
            A list of samples returned by `__getitem__`.

        Returns
        -------
        Dict[str, torch.Tensor]
            Each key contains a tensor stacked along the batch dimension.
        """
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    def tokenize_function(self, example: Mapping[str, Any]) -> Dict[str, List[int]]:
        """
        Tokenization function for use with `datasets.map`.

        Parameters
        ----------
        example : Mapping[str, Any]
            A sample or batch of samples containing the 'text' key.

        Returns
        -------
        Dict[str, List[int]]
            Output from the tokenizer (e.g., input_ids, attention_mask, token_type_ids).
        """
        return self.tokenizer(
            example['text'],
            padding='max_length',
            truncation=True,
            max_length=self.configs.max_length
        )


def get_dataloader(
    configs: omegaconf.DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: Literal["train", "valid", "test"],
) -> DataLoader[Dict[str, torch.Tensor]]:
    """
    Create a DataLoader for the given dataset split.

    Parameters
    ----------
    configs : omegaconf.DictConfig
        Configuration containing at least:
            - batch_size : int
            plus other parameters required by `IMDBDataset`.
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer used for encoding.
    split : {"train", "valid", "test"}
        Which subset to load.

    Returns
    -------
    torch.utils.data.DataLoader[Dict[str, torch.Tensor]]
        DataLoader ready for training or evaluation.
    """
    dataset = IMDBDatset(configs, tokenizer, split)
    return DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=(split=='train'),
        collate_fn=IMDBDatset.collate_fn
    )