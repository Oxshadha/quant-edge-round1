"""MODWT multiresolution analysis (Percival & Walden 2000, ch. 5) via the frequency domain.

For the MODWT, D_j = IDFT(|G_j(f)|^2 X(f)) and S_J = IDFT(|H_J(f)|^2 X(f)), where
G_j, H_J are the level-j MODWT wavelet/scaling transfer functions. Because
|G(f)|^2 + |H(f)|^2 = 1 for MODWT filters, sum_j D_j + S_J = X exactly (additive MRA).
Series are reflected before the circular transform to soften boundary effects
(Percival & Walden, sec. 5.11).
"""

from __future__ import annotations

import numpy as np
import pywt


def _transfer(filt: np.ndarray, n: int, up: int) -> np.ndarray:
    """DFT of the filter upsampled by `up` (inserting up-1 zeros), evaluated on an n-point grid."""
    f = np.arange(n) / n
    k = np.arange(len(filt))
    return np.exp(-2j * np.pi * np.outer(f * up, k)) @ filt


def modwt_mra(x: np.ndarray, wavelet: str = "sym4", level: int = 6, reflect: bool = True
              ) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (details [D1..DJ], smooth S_J), each the same length as x and summing to x."""
    x = np.asarray(x, dtype=float).ravel()
    n0 = len(x)
    xx = np.concatenate([x, x[::-1]]) if reflect else x
    n = len(xx)
    w = pywt.Wavelet(wavelet)
    h = np.asarray(w.dec_lo) / np.sqrt(2.0)
    g = np.asarray(w.dec_hi) / np.sqrt(2.0)
    X = np.fft.fft(xx)
    H_cum = np.ones(n, dtype=complex)
    details = []
    for j in range(1, level + 1):
        up = 2 ** (j - 1)
        Gj = _transfer(g, n, up) * H_cum
        details.append(np.real(np.fft.ifft(np.abs(Gj) ** 2 * X))[:n0])
        H_cum = H_cum * _transfer(h, n, up)
    smooth = np.real(np.fft.ifft(np.abs(H_cum) ** 2 * X))[:n0]
    return details, smooth


def boundary_width(wavelet: str = "sym4", level: int = 6) -> int:
    """Number of MODWT coefficients affected by the boundary at level J: (2^J - 1)(L - 1) + 1."""
    L = pywt.Wavelet(wavelet).dec_len
    return (2**level - 1) * (L - 1) + 1


def band_sum(details: list[np.ndarray], smooth: np.ndarray, levels: tuple[int, int], J: int) -> np.ndarray:
    """Sum details D_lo..D_hi (1-based); the band containing level J also gets the smooth S_J."""
    lo, hi = levels
    out = np.sum(details[lo - 1:hi], axis=0)
    if hi == J:
        out = out + smooth
    return out
