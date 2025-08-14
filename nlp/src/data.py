from typing import List, Literal, Tuple

import omegaconf
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader


class IMDBDatset(torch.utils.data.Dataset):
    def __init__(self, configs : omegaconf.DictConfig, tokenizer, split: Literal['train', 'valid', 'test']):
        """
        Inputs :
            data_config : omegaconf.DictConfig{
                model_name : str
                max_len : int
                valid_size : float
            }
            split : Literal['train', 'valid', 'test']
        Outputs : None
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
        
    def __len__(self):
        return len(self.dataset['input_ids'])

    def __getitem__(self, idx) -> Tuple[dict, int]:
        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        inputs = {key: torch.tensor(self.dataset[key][idx], dtype=torch.long) for key in self.dataset}
        
        return inputs

    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        """
        Inputs :
            batch : List[Tuple[dict, int]]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    def tokenize_function(self, example):
        return self.tokenizer(
            example['text'],
            padding='max_length',
            truncation=True,
            max_length=self.configs.max_length
        )
    
def get_dataloader(configs : omegaconf.DictConfig, tokenizer, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    """
    Output : torch.utils.data.DataLoader
    """
    dataset = IMDBDatset(configs, tokenizer, split)
    dataloader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=(split=='train'), collate_fn=IMDBDatset.collate_fn)
    return dataloader