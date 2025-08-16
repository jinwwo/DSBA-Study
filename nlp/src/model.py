from typing import Optional, Tuple

import omegaconf
import torch
import torch.nn as nn
from transformers import AutoModel


class EncoderForClassification(nn.Module):
    def __init__(self, configs : omegaconf.DictConfig) -> None:
        """
        Encoder + classification head for sequence classification.

        Parameters
        ----------
        configs : omegaconf.DictConfig
            Expected keys:
            - model_id     : str   (e.g., 'answerdotai/ModernBERT-base', 'bert-base-uncased')
            - dropout_rate : float
            - num_labels   : int

        Raises
        ------
        ValueError
            If 'model_id' is not one of the supported backbones.
        """
        super().__init__()
        
        self.model_id = configs.model_id
        self.dropout_rate = configs.dropout_rate
        self.num_labels = configs.num_labels
        
        if self.model_id not in ['answerdotai/ModernBERT-base', 'bert-base-uncased']:
            raise ValueError(f"Unsupported model_id: {self.model_id}")
        
        # model
        self.model = AutoModel.from_pretrained(self.model_id)
        self.hidden_size = self.model.config.hidden_size
        self.use_token_type_ids = (self.model_id == 'bert-base-uncased')
        
        # classification head
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass and compute logits and loss.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape: (batch_size, seq_len). Token IDs.
        attention_mask : torch.Tensor
            Shape: (batch_size, seq_len). 1 for tokens to attend, 0 for padding.
        label : torch.Tensor
            Shape: (batch_size,). Class indices (dtype: torch.long) for CrossEntropyLoss.
        token_type_ids : Optional[torch.Tensor], default=None
            Shape: (batch_size, seq_len). Segment IDs (only used for BERT).

        Returns
        -------
        logits : torch.Tensor
            Shape: (batch_size, num_labels).
        loss : torch.Tensor
            Scalar tensor with CrossEntropyLoss.
        """
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if self.use_token_type_ids and token_type_ids is not None:
            model_inputs['token_type_ids'] = token_type_ids
            
        # model outputs
        outputs = self.model(**model_inputs)

        if self.use_token_type_ids:
            #  BERT [CLS] token
            cls_input = outputs['pooler_output'] # (bsz, hidden_size)  
        else:
            token_embeddings = outputs['last_hidden_state'] # (bsz, max_seq_len, hidden_size)
            sum_embeddings = (token_embeddings * attention_mask.unsqueeze(-1)).sum(1)
            sum_mask = attention_mask.sum(1).unsqueeze(-1)
            cls_input = sum_embeddings / sum_mask # (bsz, hidden_size)
                     
        # classifier outputs
        logits = self.classifier(self.dropout(cls_input))
        
        # loss
        loss = self.criterion(logits, label)
            
        return logits, loss