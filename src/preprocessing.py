from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def preprocess_spectrum(wavelength, reflectance, degree=2, window=11, order=3):
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(reflectance, dtype=float)
    scaled = 2 * (x - x.min()) / max(np.ptp(x), np.finfo(float).eps) - 1
    baseline = np.polyval(np.polyfit(scaled, y, degree), scaled)
    corrected = y - baseline
    valid_window = min(window, len(y) if len(y) % 2 else len(y) - 1)
    if valid_window >= order + 2:
        corrected = savgol_filter(corrected, valid_window, order)
    norm = np.linalg.norm(corrected)
    return corrected / norm if norm else corrected


def matrix_from_long(spectra: pd.DataFrame, cfg: dict):
    blocks, ids = [], []
    common_axis = None
    for sample_id, g in spectra.groupby("sample_id", sort=True):
        g = g.sort_values("wavelength_nm")
        axis = g.wavelength_nm.to_numpy(float)
        if common_axis is None:
            common_axis = axis
        elif len(axis) != len(common_axis) or not np.allclose(axis, common_axis):
            raise ValueError("All spectra must use an identical wavelength grid")
        blocks.append(preprocess_spectrum(axis, g.reflectance, **cfg))
        ids.append(sample_id)
    return np.vstack(blocks), np.asarray(ids), common_axis


def augment_training(X, y, groups, rng, noise_sd=0.005):
    """Training-only augmentation; parent group IDs are preserved."""
    noisy = X + rng.normal(0, noise_sd, X.shape)
    return np.vstack([X, noisy]), np.concatenate([y, y]), np.concatenate([groups, groups])
