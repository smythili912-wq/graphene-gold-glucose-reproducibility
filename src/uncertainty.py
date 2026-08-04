from __future__ import annotations

import numpy as np


def bootstrap_metric(actual, predicted, metric, iterations=2000, alpha=0.05, seed=0):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, len(actual), len(actual))
        values.append(metric(actual[idx], predicted[idx]))
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def agreement_limits(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    bias = float(error.mean())
    sd = float(error.std(ddof=1))
    return {"bias_mM": bias, "loa_low_mM": bias - 1.96 * sd, "loa_high_mM": bias + 1.96 * sd}
