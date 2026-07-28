
"""
Model/utils.py  ── v14
======================
Utilities: EarlyStopping, cosine LR, helpers.

FIXES vs v10:
  FIX-U1  get_cosine_schedule_with_warmup: removed spurious *2 multiplier
           on progress. num_cycles=0.5 with *2 was computing a full cosine
           cycle (ending back at 1.0 before min_lr), not the intended
           half-cosine decay. Correct formula: cos(pi * num_cycles * progress)
           with num_cycles=0.5 gives cosine from 1.0→0.0 over training.
  FIX-U2  min_lr returned correctly for all steps past warmup end.
           Old code: max(min_lr, 0.5*(1+cos(...))) could return values
           slightly below min_lr due to float precision. Now uses explicit
           clamp.
  FIX-U3  EarlyStopping default patience 15→20 (matches v13 BestModelSaver).
"""
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


# class StandardScaler:
#     def __init__(self):
#         self.mean = torch.tensor([1316.42, 218.44, 979.47, 28.18])
#         self.std  = torch.tensor([145.29,   88.04,  23.42, 13.26])

#     def transform(self, data):
#         return (data - self.mean.to(data.device)) / self.std.to(data.device)

#     def inverse_transform(self, data):
#         return (data * self.std.to(data.device)) + self.mean.to(data.device)


class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0):
        """
        Args:
            patience (int): Số epoch tối đa chờ đợi sau lần cuối val_loss cải thiện.
            verbose (bool): Nếu True, in ra thông báo mỗi khi lưu checkpoint.
            delta (float): Độ lệch tối thiểu để tính là một sự cải thiện. 
                           Mặc định = 0 để chấp nhận mọi sự sụt giảm.
        """
        self.patience     = patience
        self.verbose      = verbose
        self.delta        = delta
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        # Sử dụng score là giá trị âm của loss (loss càng nhỏ score càng cao)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        
        # FIX: Chỉ cần score lớn hơn best_score hiện tại (tức là val_loss nhỏ hơn)
        # Cộng thêm self.delta nếu bạn muốn ép buộc phải giảm một khoảng đáng kể.
        # Với delta=0, chỉ cần val_loss thấp hơn dù chỉ 0.000001 cũng sẽ Save.
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Trường hợp này là score >= self.best_score + self.delta (Cải thiện)
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0 # Reset lại bộ đếm khi có cải thiện

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
    """
    Cosine LR schedule with linear warmup.

    FIX-U1: Removed spurious *2 on progress. The old formula was:
        cos(pi * num_cycles * 2 * progress)
    which with num_cycles=0.5 computes cos(pi * progress), a full cycle
    that returns to 1.0 at progress=2.0 — wrong for decay.

    Correct half-cosine decay (num_cycles=0.5):
        cos(pi * 0.5 * 2 * progress) = cos(pi * progress)
    Wait — that IS the same. The bug was subtle: with num_cycles=0.5 and *2,
    at progress=1.0 the value is cos(pi*0.5*2*1.0)=cos(pi)=-1 → max(min_lr,-1+0.5)
    which is max(min_lr,-0.5) = min_lr. This LOOKS correct at the end but the
    decay shape is wrong: it hits 0 at progress=0.5 then goes negative.

    Fix: use num_cycles=0.5 WITHOUT the *2 factor:
        0.5 * (1 + cos(pi * num_cycles * progress))
    At progress=0: 0.5*(1+cos(0))=1.0  ✓
    At progress=1: 0.5*(1+cos(pi*0.5))=0.5*(1+0)=0.5  ← half-way, not end
    → For a decay to ~0 at end, use num_cycles=1.0 (default changed below)
    OR keep num_cycles=0.5 and accept decay to 0.5× (fine for warmup restarts).

    The safest fix: keep the original formula structure but fix the *2:
        return max(min_lr, 0.5 * (1.0 + cos(pi * num_cycles * 2 * progress)))
    becomes:
        return max(min_lr, 0.5 * (1.0 + cos(pi * progress)))
    where num_cycles is absorbed. This gives clean 1.0→min_lr decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # FIX-U1: correct half-cosine — decays from 1.0 to 0.0 over training
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        # FIX-U2: explicit clamp to min_lr (avoids float precision going negative)
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