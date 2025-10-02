import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

def update_information(cfg, section, **kwargs):
    if section not in cfg:
        cfg[section] = {}
    
    cfg[section].update(kwargs)
    return cfg

def adjust_learning_rate(optimizer, epoch, lradj, learning_rate):
    # lr = learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'type3':
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate*0.1}
    elif lradj == 'type4':
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate*0.1}
    elif lradj == 'type5':
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate*0.1}
    elif lradj == 'type6':
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def check_forecasting_graph(true, predict, point, piece=1, OT_index=-1, threshold=None):
    l = true.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = np.arange(L, R)
        axs[i].plot(xticks, true[L:R, point, OT_index], color='orangered', label='true')
        axs[i].plot(xticks, predict[L:R, point, OT_index], color='midnightblue', label='predict')
        axs[i].legend()
    return fig