import numpy as np
import torch
from tqdm import tqdm
import time
import logging
import os
import glob
import json
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys

def clean_str(x):
    return str(x).replace('[', '').replace(']', '').replace("'", '').replace(',', '').strip()

def save_directory(savedir: str,
                    dataname: str,
                    modelname: str,
                    bank_name: str,
                    target: str,
                    exp_name: str):
    
    if dataname == 'bank':
        # make save directory
        bank_name_str = clean_str(bank_name)
        target_str = clean_str(target)

        savedir = os.path.join(
            savedir,
            f"{modelname}({bank_name_str}_{target_str})",
            exp_name
        )
    else:
        savedir = os.path.join(
            savedir,
            modelname,
            exp_name,
            dataname
        )
    return savedir

def version_build(accelerator, logdir: str, resume: int=None) -> str:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
    if resume is None:
        version = len(os.listdir(logdir))
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{version}')
    else:
        version = resume
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{len(os.listdir(logdir))}')
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir)
    return load_dir, save_dir

def load_resume_model(model, loaddir: str, resume: int, pre_training: bool, fine_tuning_method: str):
    new_weights = torch.load(os.path.join(loaddir,'best_model.pt'))
    
    if not pre_training:
        # List of weights to be removed
        weights_to_remove = [
            'TFTpart2.recon_layer.weight',
            'TFTpart2.recon_layer.bias',
            'TFTpart2.mlp.0.weight',
            'TFTpart2.mlp.0.bias'
        ]
        # Remove the specified weights
        for weight in weights_to_remove:
            if weight in new_weights:
                del new_weights[weight]
                
    # load weights
    model.load_state_dict(new_weights, strict=False)

    if not pre_training:
        if fine_tuning_method == 'full':
            for param in model.parameters():
                param.requires_grad = True
            print(f'Fine tuning method: Full')
        elif fine_tuning_method == 'linear_probing':
            train_params = ['final_layers']
            for name, param in model.named_parameters():
                if any(x in name for x in train_params):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f'Fine tuning method: Linear probing')
    print('load model from (version {})'.format(resume))

def prepare_data(current_loader, saved_data_dir="saved_data", status='train'):
    """
    Load or prepare data using model_key and area_grp_id extracted from current_loader.
    If data exists, load it; otherwise, process and save it.
    """
    dataset = current_loader.dataset  # Access the dataset object
    model_key = dataset.model_key  # Ensure the dataset has `model_key` attribute
    area_grp_id = dataset.area_grp_id  # Ensure the dataset has `area_grp_id` attribute
    
    # Define the directory structure
    model_dir = os.path.join(saved_data_dir, str(model_key))
    area_dir = os.path.join(model_dir, str(area_grp_id))
    os.makedirs(area_dir, exist_ok=True)
    
    # Define file paths
    x_file_path = os.path.join(area_dir, f"{status}_x.npy")
    y_file_path = os.path.join(area_dir, f"{status}_y.npy")
    
    if os.path.exists(x_file_path) and os.path.exists(y_file_path):
        print(f"Loading preprocessed data for (model_key={model_key}, area_grp_id={area_grp_id})...")
        all_x = np.load(x_file_path)
        all_y = np.load(y_file_path)
    else:
        print(f"Processing data for (model_key={model_key}, area_grp_id={area_grp_id})...")
        all_x, all_y = [], []
        total_batches = len(current_loader)
        
        for batch_idx, batch in enumerate(current_loader, start=1):
            if batch_idx % 100 == 0 or batch_idx == total_batches:
                print(f"Processing... {batch_idx}/{total_batches} ({(batch_idx / total_batches) * 100:.2f}%)")
            x, y = batch['inputs'], batch['targets']
            all_x.append(x.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
        
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        
        print(f"Saving processed data for (model_key={model_key}, area_grp_id={area_grp_id})...")
        np.save(x_file_path, all_x)
        np.save(y_file_path, all_y)
    
    return all_x, all_y

class Float32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

def progress_bar(current: int, total: int, name: str, msg: str = None, width: int = None):
    """
    progress bar to show history
    
    Parameters
    ---------
    current : int
        current batch index
    total : str
        total data length
    name : str
        description of progress bar
    msg : str(default=None)
        history of training model
    width : int(default=None)
        stty size

    """

    if width is None:
        _, term_width = os.popen('stty size', 'r').read().split()
        term_width = int(term_width)
    else:
        term_width = width

    TOTAL_BAR_LENGTH = 65.

    global last_time, begin_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(f'{name} [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = ['  Step: %s' % format_time(step_time), ' | Tot: %s' % format_time(tot_time)]
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds -= days * 3600 * 24
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    secondsf = int(seconds)
    seconds -= secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def check_graph(xs, att, ts, piece=1, threshold=None):

    ts = pd.to_datetime(ts.astype(str), format='%Y%m%d')
    l = xs.shape[0]
    chunk = (l + piece - 1) // piece
    fig, axs = plt.subplots(piece, figsize=(16, 4 * piece))

    if piece == 1:
        axs = [axs]

    for i in range(piece):
        L = i * chunk
        R = min((i + 1) * chunk, l) 

        xticks = range(L, R)

        pastel_blue = '#AEC6CF'
        pastel_red = '#FFB6B6'
        pastel_orange = '#FFDAB9'

        axs[i].plot(xticks, xs[L:R], label='Anomaly Score', color=pastel_blue, linewidth=2)

        if threshold is not None:
            axs[i].axhline(y=threshold, color=pastel_red, linestyle='--', linewidth=2, label='Threshold')

        for j in xticks:
            if att[j] == 1 and j + 1 < len(xs):
                axs[i].axvspan(j, j + 1, color=pastel_orange, alpha=0.3)
            elif att[j] == 1 and j + 1 == len(xs):
                axs[i].axvspan(j, j, color=pastel_orange, alpha=0.3)

        tick_locs = np.linspace(L, R - 1, min(6, R - L), dtype=int)
        tick_labels = ts[tick_locs].strftime('%Y-%m-%d')
        axs[i].set_xticks(tick_locs)
        axs[i].set_xticklabels(tick_labels)

        axs[i].set_title(f"Segment {i+1}: Index {L} ~ {R - 1}", fontsize=14)
        axs[i].legend()
        axs[i].set_xlabel('Timestamp', fontsize=12)
        axs[i].set_ylabel('Anomaly Score', fontsize=12)
        axs[i].grid(alpha=0.3)

    plt.tight_layout()
    return fig

def make_save(accelerator, savedir: str, resume: bool = False) -> str:
    # resume
    if resume:
        assert os.path.isdir(savedir), f'{savedir} does not exist'
        # check version
        version = len([f for f in glob.glob(os.path.join(savedir, '*')) if os.path.isdir(f)])
        # init version
        if version == 0:
            # check saved files
            files = [f for f in glob.glob(os.path.join(savedir, '*')) if os.path.isfile(f)]
            # make version0
            version0_dir = os.path.join(savedir, f'train{version}')
            
            # move saved files into version0
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs(version0_dir)
                for f in files:
                    shutil.move(f, f.replace(savedir, version0_dir))
            version += 1
        
        savedir = os.path.join(savedir, f'train{version}')

    # make save directory
    # assert not os.path.isdir(savedir), f'{savedir} already exists'
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(savedir, exist_ok=True)
    print("make save directory {}".format(savedir))

    return savedir