"""
Model/data/trajectoriesWithMe_unet.py  ── v14 (TEST / INFERENCE)
================================================================
Thin wrapper around the training dataset for test/inference use.

No changes vs v13 wrapper logic — inherits all v13/v14 fixes from
trajectoriesWithMe_unet_training.py:
  FIX-DATA-1..4, FIX-CACHE-1 (all in training module).

The class simply forces is_test=True and type="test" defaults so
callers don't need to specify them explicitly.
"""
from __future__ import annotations

from Model.data.trajectoriesWithMe_unet_training import (
    TrajectoryDataset as _TrainingDataset,
    seq_collate,
    env_data_processing,
)


class TrajectoryDataset(_TrainingDataset):
    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim=" ",
        other_modal="gph",
        test_year=None,
        type="test",
        is_test=True,
        **kwargs,
    ):
        super().__init__(
            data_dir     = data_dir,
            obs_len      = obs_len,
            pred_len     = pred_len,
            skip         = skip,
            threshold    = threshold,
            min_ped      = min_ped,
            delim        = delim,
            other_modal  = other_modal,
            test_year    = test_year,
            type         = type,
            is_test      = is_test,
            **kwargs,
        )


__all__ = ["TrajectoryDataset", "seq_collate", "env_data_processing"]