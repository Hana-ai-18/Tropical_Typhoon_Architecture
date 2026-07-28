from __future__ import annotations

import os
import time
import math
from contextlib import contextmanager

import torch
import numpy as np


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def bool_flag(s):
    return s == '1'


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def dic2cuda(env_data, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key in env_data:
        if torch.is_tensor(env_data[key]):
            env_data[key] = env_data[key].to(device)
        else:
            env_data[key] = torch.tensor(env_data[key], dtype=torch.float).to(device)
    return env_data



class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0):
       
        self.patience     = patience
        self.verbose      = verbose
        self.delta        = delta
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
      
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
           
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0 

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.val_loss_min == np.inf:
                print(f'Initial validation loss: {val_loss:.6f}. Saving model...')
            else:
                print(f'Val loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(), 
            'val_loss': val_loss
        }, os.path.join(path, 'best_model.pth'))
        
        self.val_loss_min = val_loss

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 1e-7,
):
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )   
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr, cosine_val)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_dset_path(dset_name, dset_type):
    return {'root': dset_name, 'type': dset_type}


def relative_to_abs(rel_traj, start_pos):
    return torch.cumsum(rel_traj, dim=0) + start_pos.unsqueeze(0)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@contextmanager
def timeit(msg, should_time=True):
    if should_time and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    yield
    if should_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f'{msg}: {(time.time() - t0) * 1000.0:.2f} ms')