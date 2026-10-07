"""MODWT-MRA of standardized residuals -> horizon-band matrices H1/H2/H3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BANDS, J_LEVELS, WAVELET
from src.modwt import band_sum, boundary_width, modwt_mra


def decompose(z: pd.DataFrame | np.ndarray, trim: bool = True) -> dict[str, np.ndarray]:
    """Return {band: (n_kept x d) array}. With trim=True, boundary-affected rows at both ends are dropped."""
    arr = np.asarray(z, dtype=float)
    n, d = arr.shape
    out = {b: np.empty((n, d)) for b in BANDS}
    for k in range(d):
        details, smooth = modwt_mra(arr[:, k], WAVELET, J_LEVELS)
        for b, lv in BANDS.items():
            out[b][:, k] = band_sum(details, smooth, lv, J_LEVELS)
    if trim:
        L = min(boundary_width(WAVELET, J_LEVELS), n // 4)
        out = {b: m[L:n - L] for b, m in out.items()}
    return out


def decompose_trailing(z: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
    """For rolling forecasts: keep the full window (reflection boundary) so the most recent data count."""
    return decompose(z, trim=False)
