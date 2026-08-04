from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fit_calibration(df: pd.DataFrame) -> dict:
    used = df.dropna(subset=["shift_nm", "concentration_mM"])
    slope, intercept, r, p, slope_se = stats.linregress(used.concentration_mM, used.shift_nm)
    pred = intercept + slope * used.concentration_mM.to_numpy()
    residual = used.shift_nm.to_numpy() - pred
    blanks = used.loc[used.concentration_mM == 0, "shift_nm"].to_numpy()
    sigma_blank = float(np.std(blanks, ddof=1)) if len(blanks) > 1 else np.nan
    abs_slope = abs(float(slope))
    return {"n": len(used), "n_blanks": len(blanks), "slope_nm_per_mM": float(slope),
            "intercept_nm": float(intercept), "r2": float(r * r), "p_value": float(p),
            "slope_se": float(slope_se), "rmse_nm": float(np.sqrt(np.mean(residual ** 2))),
            "lod_mM": 3 * sigma_blank / abs_slope if abs_slope else np.nan,
            "loq_mM": 10 * sigma_blank / abs_slope if abs_slope else np.nan}


def cluster_bootstrap(df: pd.DataFrame, iterations: int, alpha: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    batches = df.batch_id.unique()
    slopes = []
    if len(batches) < 2:
        return {"slope_ci_low": np.nan, "slope_ci_high": np.nan}
    for _ in range(iterations):
        sampled = rng.choice(batches, len(batches), replace=True)
        chunks = [df[df.batch_id == batch].assign(_boot=i) for i, batch in enumerate(sampled)]
        boot = pd.concat(chunks, ignore_index=True)
        slopes.append(stats.linregress(boot.concentration_mM, boot.shift_nm).slope)
    low, high = np.quantile(slopes, [alpha / 2, 1 - alpha / 2])
    return {"slope_ci_low": float(low), "slope_ci_high": float(high)}
