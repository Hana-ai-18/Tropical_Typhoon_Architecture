from __future__ import annotations

import os
import sys

# Ensure project root is on path regardless of working directory
_here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _here not in sys.path:
    sys.path.insert(0, _here)

from torch.utils.data import DataLoader

from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate


def _find_tcnd_root(path: str) -> str:

    path  = os.path.abspath(path)
    check = path
    for _ in range(6):
        if os.path.exists(os.path.join(check, "Data1d")):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent

  
    for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
        candidate = os.path.join(path, sub)
        if os.path.exists(os.path.join(candidate, "Data1d")):
            return candidate
 
    return path

def data_loader(
    args,
    path_config,
    test:      bool     = False,
    test_year: int | None = None,
    batch_size: int | None = None,
) -> tuple:

    if isinstance(path_config, dict):
        raw_path  = path_config.get("root", "")
        dset_type = path_config.get("type", "test" if test else "train")
    else:
        raw_path  = str(path_config)
        dset_type = "test" if test else "train"

    root = _find_tcnd_root(raw_path)
    print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

    dataset = TrajectoryDataset(
        data_dir    = root,
        obs_len     = args.obs_len,
        pred_len    = args.pred_len,
        skip        = getattr(args, "skip",        1),
        threshold   = getattr(args, "threshold",   0.002),
        min_ped     = getattr(args, "min_ped",      1),
        delim       = getattr(args, "delim",        " "),
        other_modal = getattr(args, "other_modal",  "gph"),
        test_year   = test_year,
        # type        = dset_type,
        split=dset_type,
        is_test     = test,
        filter_region   = getattr(args, "filter_region",   False),
        min_pct_in_scs  = getattr(args, "min_pct_in_scs",  15.0),
    )

    num_workers = getattr(args, "num_workers", 0)

    loader = DataLoader(
        dataset,
        batch_size  = batch_size or args.batch_size,
        shuffle     = not test,
        collate_fn  = seq_collate,
        num_workers = num_workers,
        drop_last   = False,
        pin_memory  = _cuda_available(),
    )

    print(f"  {len(dataset)} sequences loaded")
    return dataset, loader


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False