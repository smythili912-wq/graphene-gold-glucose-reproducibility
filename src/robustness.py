from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def stress_test(X, y, cfg: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(X, y)
    rows = []
    for sd in cfg["noise_sd"]:
        perturbed = X + rng.normal(0, sd, X.shape)
        rows.append({"perturbation": "gaussian_noise", "magnitude": sd,
                     "rmse_mM": mean_squared_error(y, model.predict(perturbed)) ** 0.5})
    grid = np.linspace(-1, 1, X.shape[1])
    for slope in cfg["baseline_slope"]:
        perturbed = X + slope * grid
        rows.append({"perturbation": "baseline_slope", "magnitude": slope,
                     "rmse_mM": mean_squared_error(y, model.predict(perturbed)) ** 0.5})
    for shift in cfg["wavelength_shift_nm"]:
        columns = max(1, int(round(abs(shift))))
        perturbed = np.roll(X, columns if shift > 0 else -columns, axis=1)
        rows.append({"perturbation": "wavelength_shift_nm", "magnitude": shift,
                     "rmse_mM": mean_squared_error(y, model.predict(perturbed)) ** 0.5})
    return pd.DataFrame(rows)
