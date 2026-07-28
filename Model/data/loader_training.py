# """
# Model/data/loader_training.py  ── v11
# ======================================
# Data loader for training.

# FIXES từ v10-fixed-2 (giữ nguyên):
#   1. REMOVED _normalize_data() — Data1d files already normalised.
#   2. seq_collate env_data tại index 13.
#   3. Removed unused TCTrainingDataset.
#   4. prefetch_factor=None khi num_workers=0.
#   5. persistent_workers=False khi num_workers=0.
#   6. 'type' → 'split' kwarg cho TrajectoryDataset.
#   7. _find_tcnd_root check path trước khi walk up.
#   8. pin_memory guard: chỉ True khi CUDA available AND num_workers > 0.
#   9. test_year forwarded chỉ khi dataset constructor chấp nhận.

# FIXES mới v11:
#   10. BUG-3 FIX: import từ trajectoriesWithMe_unet_training (v11 dataset).
#       Dùng split= kwarg (không phải type=) nhất quán.
#   11. Thêm worker_init_fn để seed numpy/random per worker → reproducible.
#   12. Thêm drop_last=True cho train split để tránh batch size 1 ở cuối.
# """
# from __future__ import annotations

# import inspect
# import os
# import random

# import numpy as np
# import torch
# from torch.utils.data import DataLoader

# # BUG-3 FIX: import từ v11 training dataset (dùng split= kwarg)
# from Model.data.trajectoriesWithMe_unet_training import (
#     TrajectoryDataset,
#     seq_collate,
# )


# def _find_tcnd_root(path: str) -> str:
#     """
#     Walk up the directory tree to find the folder that contains Data1d/.
#     FIX-7: check `path` itself trước khi ascend.
#     """
#     path = os.path.abspath(path)

#     # Check given path first
#     if os.path.exists(os.path.join(path, "Data1d")):
#         return path

#     # Walk upward
#     check = os.path.dirname(path)
#     for _ in range(6):
#         if os.path.exists(os.path.join(check, "Data1d")):
#             return check
#         parent = os.path.dirname(check)
#         if parent == check:
#             break
#         check = parent

#     # Scan well-known sub-directory names
#     for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
#         candidate = os.path.join(path, sub)
#         if os.path.exists(os.path.join(candidate, "Data1d")):
#             return candidate

#     return path


# def _worker_init_fn(worker_id: int) -> None:
#     """Seed numpy/random per DataLoader worker for reproducibility."""
#     seed = torch.initial_seed() % (2 ** 32)
#     np.random.seed(seed + worker_id)
#     random.seed(seed + worker_id)


# def _cuda_available() -> bool:
#     try:
#         import torch as _torch
#         return _torch.cuda.is_available()
#     except ImportError:
#         return False


# def data_loader(
#     args,
#     path_config,
#     test: bool = False,
#     test_year=None,
#     batch_size: int | None = None,
#     seed: int | None = None,
# ):
#     """
#     Unified data loader for train / val / test splits.

#     path_config : str  or  {"root": ..., "type": "train"|"val"|"test"}

#     BUG-3 FIX: TrajectoryDataset v11 uses split= kwarg; this loader
#     always passes split=dset_type (never the old type= kwarg) except when
#     the dataset signature only has 'type' (legacy fallback).
#     """
#     if isinstance(path_config, dict):
#         raw_path  = path_config.get("root", "")
#         dset_type = path_config.get("type", "test" if test else "train")
#     else:
#         raw_path  = str(path_config)
#         dset_type = "test" if test else "train"

#     root = _find_tcnd_root(raw_path)
#     print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

#     # FIX-9: only forward test_year if constructor accepts it
#     # FIX-6 / BUG-3: use split= kwarg preferentially
#     ds_sig    = inspect.signature(TrajectoryDataset.__init__)
#     ds_kwargs: dict = dict(
#         data_dir    = root,
#         obs_len     = args.obs_len,
#         pred_len    = args.pred_len,
#         skip        = getattr(args, "skip",        1),
#         threshold   = getattr(args, "threshold",   0.002),
#         min_ped     = getattr(args, "min_ped",     1),
#         delim       = getattr(args, "delim",       " "),
#         other_modal = getattr(args, "other_modal", "gph"),
#         is_test     = test,
#     )

#     # v11 dataset uses 'split', legacy uses 'type'
#     if "split" in ds_sig.parameters:
#         ds_kwargs["split"] = dset_type
#     elif "type" in ds_sig.parameters:
#         ds_kwargs["type"] = dset_type
#     else:
#         ds_kwargs["split"] = dset_type  # default to split

#     if "test_year" in ds_sig.parameters and test_year is not None:
#         ds_kwargs["test_year"] = test_year

#     dataset = TrajectoryDataset(**ds_kwargs)

#     num_workers = getattr(args, "num_workers", 0)

#     # FIX-4+5: guard both persistent_workers and prefetch_factor
#     use_persistent = num_workers > 0
#     prefetch       = 2 if num_workers > 0 else None

#     # FIX-8: pin_memory only useful with CUDA + background workers
#     use_pin_memory = _cuda_available() and num_workers > 0

#     # drop_last=True for train split to avoid single-sample batches
#     # that break BatchNorm in FNO layers
#     is_train   = dset_type == "train" and not test
#     drop_last  = is_train and len(dataset) > (batch_size or args.batch_size)

#     # Dedicated generator so shuffle order is independent of the global torch
#     # RNG that the model consumes (torch.rand/randn in CFM sampling, augment).
#     # Without this, batch order per epoch drifts with how many times the model
#     # touched the global RNG → not reproducible across runs / seeds.
#     # Falls back to args.seed if seed arg not passed.
#     _seed = seed if seed is not None else getattr(args, "seed", None)
#     gen = None
#     if _seed is not None and not test:
#         gen = torch.Generator()
#         gen.manual_seed(int(_seed))

#     loader = DataLoader(
#         dataset,
#         batch_size         = batch_size or args.batch_size,
#         shuffle            = not test,
#         collate_fn         = seq_collate,
#         num_workers        = num_workers,
#         persistent_workers = use_persistent,
#         prefetch_factor    = prefetch,
#         drop_last          = drop_last,
#         pin_memory         = use_pin_memory,
#         worker_init_fn     = _worker_init_fn if num_workers > 0 else None,
#         generator          = gen,
#     )
#     print(f"  {len(dataset)} sequences  "
#           f"(workers={num_workers}, drop_last={drop_last})")
#     return dataset, loader
"""
Model/data/loader_training.py  ── v11
======================================
Data loader for training.

FIXES từ v10-fixed-2 (giữ nguyên):
  1. REMOVED _normalize_data() — Data1d files already normalised.
  2. seq_collate env_data tại index 13.
  3. Removed unused TCTrainingDataset.
  4. prefetch_factor=None khi num_workers=0.
  5. persistent_workers=False khi num_workers=0.
  6. 'type' → 'split' kwarg cho TrajectoryDataset.
  7. _find_tcnd_root check path trước khi walk up.
  8. pin_memory guard: chỉ True khi CUDA available AND num_workers > 0.
  9. test_year forwarded chỉ khi dataset constructor chấp nhận.

FIXES mới v11:
  10. BUG-3 FIX: import từ trajectoriesWithMe_unet_training (v11 dataset).
      Dùng split= kwarg (không phải type=) nhất quán.
  11. Thêm worker_init_fn để seed numpy/random per worker → reproducible.
  12. Thêm drop_last=True cho train split để tránh batch size 1 ở cuối.
"""
from __future__ import annotations

import inspect
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

# BUG-3 FIX: import từ v11 training dataset (dùng split= kwarg)
from Model.data.trajectoriesWithMe_unet_training import (
    TrajectoryDataset,
    seq_collate,
)


def _find_tcnd_root(path: str) -> str:
    """
    Walk up the directory tree to find the folder that contains Data1d/.
    FIX-7: check `path` itself trước khi ascend.
    """
    path = os.path.abspath(path)

    # Check given path first
    if os.path.exists(os.path.join(path, "Data1d")):
        return path

    # Walk upward
    check = os.path.dirname(path)
    for _ in range(6):
        if os.path.exists(os.path.join(check, "Data1d")):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent

    # Scan well-known sub-directory names
    for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
        candidate = os.path.join(path, sub)
        if os.path.exists(os.path.join(candidate, "Data1d")):
            return candidate

    return path


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy/random per DataLoader worker for reproducibility."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _cuda_available() -> bool:
    try:
        import torch as _torch
        return _torch.cuda.is_available()
    except ImportError:
        return False


def data_loader(
    args,
    path_config,
    test: bool = False,
    test_year=None,
    batch_size: int | None = None,
    seed: int | None = None,
):
    """
    Unified data loader for train / val / test splits.

    path_config : str  or  {"root": ..., "type": "train"|"val"|"test"}

    BUG-3 FIX: TrajectoryDataset v11 uses split= kwarg; this loader
    always passes split=dset_type (never the old type= kwarg) except when
    the dataset signature only has 'type' (legacy fallback).
    """
    if isinstance(path_config, dict):
        raw_path  = path_config.get("root", "")
        dset_type = path_config.get("type", "test" if test else "train")
    else:
        raw_path  = str(path_config)
        dset_type = "test" if test else "train"

    root = _find_tcnd_root(raw_path)
    print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

    # FIX-9: only forward test_year if constructor accepts it
    # FIX-6 / BUG-3: use split= kwarg preferentially
    ds_sig    = inspect.signature(TrajectoryDataset.__init__)
    ds_kwargs: dict = dict(
        data_dir    = root,
        obs_len     = args.obs_len,
        pred_len    = args.pred_len,
        skip        = getattr(args, "skip",        1),
        threshold   = getattr(args, "threshold",   0.002),
        min_ped     = getattr(args, "min_ped",     1),
        delim       = getattr(args, "delim",       " "),
        other_modal = getattr(args, "other_modal", "gph"),
        is_test     = test,
    )

    # v11 dataset uses 'split', legacy uses 'type'
    if "split" in ds_sig.parameters:
        ds_kwargs["split"] = dset_type
    elif "type" in ds_sig.parameters:
        ds_kwargs["type"] = dset_type
    else:
        ds_kwargs["split"] = dset_type  # default to split

    if "test_year" in ds_sig.parameters and test_year is not None:
        ds_kwargs["test_year"] = test_year

    # [FIX-DATA-30] Optional region filter — same defensive pattern as
    # test_year above: only forward if the dataset constructor accepts it.
    if "filter_region" in ds_sig.parameters:
        ds_kwargs["filter_region"] = getattr(args, "filter_region", False)
    if "min_pct_in_scs" in ds_sig.parameters:
        ds_kwargs["min_pct_in_scs"] = getattr(args, "min_pct_in_scs", 15.0)

    dataset = TrajectoryDataset(**ds_kwargs)

    num_workers = getattr(args, "num_workers", 0)

    # FIX-4+5: guard both persistent_workers and prefetch_factor
    use_persistent = num_workers > 0
    prefetch       = 2 if num_workers > 0 else None

    # FIX-8: pin_memory only useful with CUDA + background workers
    use_pin_memory = _cuda_available() and num_workers > 0

    # drop_last=True for train split to avoid single-sample batches
    # that break BatchNorm in FNO layers
    is_train   = dset_type == "train" and not test
    drop_last  = is_train and len(dataset) > (batch_size or args.batch_size)

    # Dedicated generator so shuffle order is independent of the global torch
    # RNG that the model consumes (torch.rand/randn in CFM sampling, augment).
    # Without this, batch order per epoch drifts with how many times the model
    # touched the global RNG → not reproducible across runs / seeds.
    # Falls back to args.seed if seed arg not passed.
    _seed = seed if seed is not None else getattr(args, "seed", None)
    gen = None
    if _seed is not None and not test:
        gen = torch.Generator()
        gen.manual_seed(int(_seed))

    loader = DataLoader(
        dataset,
        batch_size         = batch_size or args.batch_size,
        shuffle            = not test,
        collate_fn         = seq_collate,
        num_workers        = num_workers,
        persistent_workers = use_persistent,
        prefetch_factor    = prefetch,
        drop_last          = drop_last,
        pin_memory         = use_pin_memory,
        worker_init_fn     = _worker_init_fn if num_workers > 0 else None,
        generator          = gen,
    )
    print(f"  {len(dataset)} sequences  "
          f"(workers={num_workers}, drop_last={drop_last})")
    return dataset, loader