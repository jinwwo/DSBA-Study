from typing import Optional, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EncoderForClassification(nn.Module):
    def __init__(self, configs : omegaconf.DictConfig):
        super(EncoderForClassification, self).__init__()
        
        self.model_id = configs.model_id
        self.dropout_rate = configs.dropout_rate
        self.num_labels = configs.num_labels
        
        if self.model_id not in ['answerdotai/ModernBERT-base', 'bert-base-uncased']:
            raise ValueError(f"Unsupported model_id: {self.model_id}")
            
        # model
        self.model = AutoModel.from_pretrained(self.model_id)
        self.hidden_size = self.model.config.hidden_size
        self.use_token_type_ids = self.model_id == 'bert-base-uncased'
        
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
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
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