import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder LSTM for autoregressive prediction
        self.decoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.projection = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, seq_len, enc_in] - encoder input
        # x_dec: [B, label_len + pred_len, enc_in] - decoder input
        
        batch_size = x_enc.shape[0]
        
        # Encode the input sequence
        _, (hidden, cell) = self.encoder(x_enc)
        
        # Prepare decoder input (use label_len from x_dec for teacher forcing)
        if self.training:
            # Teacher forcing during training
            decoder_input = x_dec[:, :self.label_len + self.pred_len - 1, :]  # [B, L+S-1, D]
            decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
            predictions = self.projection(decoder_out)  # [B, L+S-1, D]
            # Return only the prediction part
            return predictions[:, self.label_len-1:, :]  # [B, S, D]
        else:
            # Autoregressive inference
            outputs = []
            decoder_input = x_dec[:, :self.label_len, :]  # Start with label sequence
            
            # Get initial decoder states
            decoder_out, (h_dec, c_dec) = self.decoder(decoder_input, (hidden, cell))
            
            # Use last output for first prediction
            current_input = decoder_out[:, -1:, :]  # [B, 1, H]
            pred = self.projection(current_input)  # [B, 1, D]
            outputs.append(pred)
            
            # Autoregressive generation
            for t in range(1, self.pred_len):
                decoder_out, (h_dec, c_dec) = self.decoder(pred, (h_dec, c_dec))
                pred = self.projection(decoder_out)  # [B, 1, D]
                outputs.append(pred)
            
            return torch.cat(outputs, dim=1)  # [B, S, D]