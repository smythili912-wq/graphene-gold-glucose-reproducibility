from __future__ import annotations

import numpy as np
import pandas as pd


def resonance_features(spectra: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id, g in spectra.groupby("sample_id", sort=True):
        g = g.sort_values("wavelength_nm")
        x, y = g.wavelength_nm.to_numpy(float), g.reflectance.to_numpy(float)
        idx = int(np.argmin(y))
        resonance = float(x[idx])
        depth = float(np.median(y) - y[idx])
        half = y[idx] + depth / 2
        below = np.flatnonzero(y <= half)
        fwhm = float(x[below[-1]] - x[below[0]]) if len(below) > 1 else np.nan
        rows.append({"sample_id": sample_id, "resonance_nm": resonance,
                     "depth": depth, "fwhm_nm": fwhm,
                     "q_factor": resonance / fwhm if fwhm > 0 else np.nan})
    return pd.DataFrame(rows)


def add_shift(features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.merge(features, on="sample_id", validate="one_to_one")
    blank = out[out.concentration_mM == 0].groupby("batch_id").resonance_nm.mean()
    out["blank_resonance_nm"] = out.batch_id.map(blank)
    out["shift_nm"] = out.resonance_nm - out.blank_resonance_nm
    return out
