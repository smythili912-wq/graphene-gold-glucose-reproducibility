from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_validation import assert_group_disjoint


def models(seed: int, n_features: int):
    return {
        "linear": make_pipeline(StandardScaler(), PLSRegression(n_components=min(8, n_features))),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                random_state=seed, n_jobs=-1),
    }


def grouped_evaluate(X, y, groups, folds: int, seed: int):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two independent groups are required")
    splitter = GroupKFold(n_splits=min(folds, len(unique)))
    rows, predictions = [], []
    for name, estimator in models(seed, X.shape[1]).items():
        for fold, (train, test) in enumerate(splitter.split(X, y, groups), 1):
            assert_group_disjoint(groups[train], groups[test])
            fitted = clone(estimator).fit(X[train], y[train])
            pred = np.asarray(fitted.predict(X[test])).ravel()
            rows.append({"model": name, "fold": fold,
                         "rmse_mM": mean_squared_error(y[test], pred) ** 0.5,
                         "mae_mM": mean_absolute_error(y[test], pred),
                         "r2": r2_score(y[test], pred) if len(test) > 1 else np.nan,
                         "n_test": len(test)})
            predictions.extend({"model": name, "fold": fold, "actual_mM": float(a),
                                "predicted_mM": float(p), "group": str(g)}
                               for a, p, g in zip(y[test], pred, groups[test]))
    return pd.DataFrame(rows), pd.DataFrame(predictions)
