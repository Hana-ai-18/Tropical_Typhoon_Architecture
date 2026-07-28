# """
# TCNM/data/loader.py  ── v9-fixed
# ==================================
# Smart data loader — resolves TCND_vn root automatically.
# Compatible with Kaggle (Google Drive mount) and local paths.

# Kaggle + Drive usage:
#     from google.colab import drive
#     drive.mount('/content/drive')

#     # In train script:
#     --dataset_root /content/drive/MyDrive/TCND_vn

#     # Or pure Kaggle input:
#     --dataset_root /kaggle/input/tcnd-vn/TCND_vn

# The loader walks up the directory tree to find the folder containing Data1d/.
# """
# from __future__ import annotations

# import os
# import sys

# # Ensure project root is on path regardless of working directory
# _here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if _here not in sys.path:
#     sys.path.insert(0, _here)

# from torch.utils.data import DataLoader

# from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate


# def _find_tcnd_root(path: str) -> str:
#     """
#     Walk up the directory tree to find the folder containing Data1d/.
#     Also checks common sub-folder names used in Kaggle/Drive setups.
#     """
#     path  = os.path.abspath(path)
#     check = path
#     for _ in range(6):
#         if os.path.exists(os.path.join(check, "Data1d")):
#             return check
#         parent = os.path.dirname(check)
#         if parent == check:
#             break
#         check = parent

#     # Try common sub-folder names
#     for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
#         candidate = os.path.join(path, sub)
#         if os.path.exists(os.path.join(candidate, "Data1d")):
#             return candidate

#     # Fallback: use path as-is (will raise clear error from dataset)
#     return path



#     """
#     Create a DataLoader for the given split.

#     Parameters
#     ----------
#     args        : argparse Namespace with obs_len, pred_len, batch_size, etc.
#     path_config : str | dict — {'root': ..., 'type': 'train'|'val'|'test'}
#     test        : bool — if True, shuffle=False and dtype defaults to 'test'
#     test_year   : int | None — filter Data1d files by year string
#     batch_size  : override args.batch_size if provided

#     Returns
#     -------
#     (dataset, loader)
#     """
# def data_loader(
#     args,
#     path_config,
#     test:      bool     = False,
#     test_year: int | None = None,
#     batch_size: int | None = None,
# ) -> tuple:

#     if isinstance(path_config, dict):
#         raw_path  = path_config.get("root", "")
#         dset_type = path_config.get("type", "test" if test else "train")
#     else:
#         raw_path  = str(path_config)
#         dset_type = "test" if test else "train"

#     root = _find_tcnd_root(raw_path)
#     print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

#     dataset = TrajectoryDataset(
#         data_dir    = root,
#         obs_len     = args.obs_len,
#         pred_len    = args.pred_len,
#         skip        = getattr(args, "skip",        1),
#         threshold   = getattr(args, "threshold",   0.002),
#         min_ped     = getattr(args, "min_ped",      1),
#         delim       = getattr(args, "delim",        " "),
#         other_modal = getattr(args, "other_modal",  "gph"),
#         test_year   = test_year,
#         # type        = dset_type,
#         split=dset_type,
#         is_test     = test,
#     )

#     # num_workers: 0 for Kaggle/Colab stability; increase locally if fast SSD
#     num_workers = getattr(args, "num_workers", 0)

#     loader = DataLoader(
#         dataset,
#         batch_size  = batch_size or args.batch_size,
#         shuffle     = not test,
#         collate_fn  = seq_collate,
#         num_workers = num_workers,
#         drop_last   = False,
#         pin_memory  = _cuda_available(),
#     )

#     print(f"  {len(dataset)} sequences loaded")
#     return dataset, loader


# def _cuda_available() -> bool:
#     try:
#         import torch
#         return torch.cuda.is_available()
#     except ImportError:
#         return False
"""
TCNM/data/loader.py  ── v9-fixed
==================================
Smart data loader — resolves TCND_vn root automatically.
Compatible with Kaggle (Google Drive mount) and local paths.

Kaggle + Drive usage:
    from google.colab import drive
    drive.mount('/content/drive')

    # In train script:
    --dataset_root /content/drive/MyDrive/TCND_vn

    # Or pure Kaggle input:
    --dataset_root /kaggle/input/tcnd-vn/TCND_vn

The loader walks up the directory tree to find the folder containing Data1d/.
"""
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
    """
    Walk up the directory tree to find the folder containing Data1d/.
    Also checks common sub-folder names used in Kaggle/Drive setups.
    """
    path  = os.path.abspath(path)
    check = path
    for _ in range(6):
        if os.path.exists(os.path.join(check, "Data1d")):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent

    # Try common sub-folder names
    for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
        candidate = os.path.join(path, sub)
        if os.path.exists(os.path.join(candidate, "Data1d")):
            return candidate

    # Fallback: use path as-is (will raise clear error from dataset)
    return path



    """
    Create a DataLoader for the given split.

    Parameters
    ----------
    args        : argparse Namespace with obs_len, pred_len, batch_size, etc.
    path_config : str | dict — {'root': ..., 'type': 'train'|'val'|'test'}
    test        : bool — if True, shuffle=False and dtype defaults to 'test'
    test_year   : int | None — filter Data1d files by year string
    batch_size  : override args.batch_size if provided

    Returns
    -------
    (dataset, loader)
    """
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
        # [FIX-DATA-30] Optional region filter — keep only storms whose
        # track has >= min_pct_in_scs% of points in the South China Sea
        # / Vietnam region. Off by default; pass --filter_region to
        # enable via CLI.
        filter_region   = getattr(args, "filter_region",   False),
        min_pct_in_scs  = getattr(args, "min_pct_in_scs",  15.0),
    )

    # num_workers: 0 for Kaggle/Colab stability; increase locally if fast SSD
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