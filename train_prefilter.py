"""
CPU-side entry prefilter (matches backtester lookback=1 + signal threshold).

Used in the main process only (not in worker processes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_signal_bool_matrix(
    train_df: pd.DataFrame,
    keys: list[str],
    signals_dict: dict[str, dict[str, Any]],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for k in keys:
        sig = signals_dict[k]["fn"](train_df)
        if not isinstance(sig, pd.Series):
            sig = pd.Series(False, index=train_df.index)
        rows.append(sig.fillna(False).to_numpy(dtype=np.bool_, copy=False))
    return np.stack(rows, axis=0)


def numpy_entry_prefilter(
    strategies: list,
    sig_mat: np.ndarray,
    key_to_idx: dict[str, int],
    pre_batch: int = 2048,
) -> tuple[list, list[tuple[int, dict]]]:
    """Return (raw_results with None placeholders, pending list of (global_idx, strat))."""
    raw_results: list = [None] * len(strategies)
    pending: list[tuple[int, dict]] = []

    for batch_start in range(0, len(strategies), pre_batch):
        batch = strategies[batch_start : batch_start + pre_batch]
        Bb = len(batch)
        max_n = max((len(s["signals"]) for s in batch), default=0)
        if max_n == 0:
            for j in range(Bb):
                raw_results[batch_start + j] = None
            continue

        idx_mat = np.zeros((Bb, max_n), dtype=np.int64)
        valid_m = np.zeros((Bb, max_n), dtype=np.bool_)
        thresh = np.zeros((Bb, 1), dtype=np.int64)

        for i, s in enumerate(batch):
            sigs = s["signals"]
            n_s = len(sigs)
            thresh[i, 0] = max(2, int(n_s * 0.75))
            if not sigs:
                continue
            if not all(kk in key_to_idx for kk in sigs):
                continue
            for j, kk in enumerate(sigs):
                idx_mat[i, j] = key_to_idx[kk]
                valid_m[i, j] = True

        gathered = sig_mat[idx_mat]
        counts = (gathered & valid_m[:, :, np.newaxis]).sum(axis=1)
        ever = (counts >= thresh).any(axis=1)

        for j in range(Bb):
            gi = batch_start + j
            s = batch[j]
            sigs = s["signals"]
            if not sigs:
                raw_results[gi] = None
                continue
            if not all(kk in key_to_idx for kk in sigs):
                pending.append((gi, s))
                continue
            if not ever[j]:
                raw_results[gi] = None
            else:
                pending.append((gi, s))

    return raw_results, pending
