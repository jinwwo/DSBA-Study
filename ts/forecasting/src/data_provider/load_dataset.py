from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features_from_date


def load_dataset(
    datadir: str,
    dataname: str,
    split_rate: list,
    time_embedding: list = [True, 'h'],
    del_feature: list = None,
    seed: int = 42
):
    df = pd.read_csv(os.path.join(datadir, f"{dataname}.csv"))
    df_ts = df[['date']]
    df = df.drop(columns=['date'])
    var = len(df.columns)
    
    train_size = int(len(df) * split_rate[0])
    val_size = int(len(df) * split_rate[1])

    train, val, test = np.split(df.values, [train_size, train_size + val_size])

    train_ts = df_ts[:train_size]
    val_ts = df_ts[train_size:train_size + val_size]
    test_ts = df_ts[train_size + val_size:]

    return train, train_ts, val, val_ts, test, test_ts, var