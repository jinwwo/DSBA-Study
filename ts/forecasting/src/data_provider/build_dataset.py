from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd


class BuildDataset(Dataset):
    def __init__(self, data, ts, seq_len, label_len, pred_len, time_embedding):
        self.data = np.array(data)   # [N, D]
        self.ts = pd.to_datetime(ts['date'])
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.time_embedding = time_embedding

        # 간단히 month만 time embedding으로 사용
        self.ts = self.ts.dt.month.astype('int16').to_numpy()

        # 가능한 샘플 개수
        self.valid_window = len(self.data) - (self.seq_len + self.pred_len) + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        # ----------------------
        # Encoder 입력
        # ----------------------
        x_enc = torch.as_tensor(self.data[s_begin:s_end]).float()           # [seq_len, D]
        x_mark_enc = torch.as_tensor(self.ts[s_begin:s_end]).long()         # [seq_len]

        # ----------------------
        # Decoder 입력 (label_len + pred_len 길이)
        # ----------------------
        x_dec = torch.as_tensor(self.data[r_begin:r_end]).float()           # [label_len+pred_len, D]
        x_mark_dec = torch.as_tensor(self.ts[r_begin:r_end]).long()         # [label_len+pred_len]

        # ----------------------
        # Ground truth
        # ----------------------
        y = torch.as_tensor(self.data[s_end:r_end]).float()                 # [pred_len, D]

        return x_enc, x_mark_enc, x_dec, x_mark_dec, y

    def __len__(self):
        return self.valid_window
