from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/graphene_glucose_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figures(calibration_df, predictions, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    summary = calibration_df.groupby("concentration_mM").shift_nm.agg(["mean", "std", "count"])
    sem95 = 1.96 * summary["std"] / np.sqrt(summary["count"])
    ax.errorbar(summary.index, summary["mean"], yerr=sem95, fmt="o", capsize=3)
    ax.set(xlabel="Glucose concentration (mM)", ylabel="Resonance shift (nm)",
           title="Calibration with approximate 95% intervals")
    fig.tight_layout(); fig.savefig(outdir / "calibration.png", dpi=220); plt.close(fig)

    best = predictions[predictions.model == "ridge"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(best.actual_mM, best.predicted_mM, alpha=.7)
    limits = [min(best.actual_mM.min(), best.predicted_mM.min()),
              max(best.actual_mM.max(), best.predicted_mM.max())]
    ax.plot(limits, limits, "--", color="black", linewidth=1)
    ax.set(xlabel="Reference concentration (mM)", ylabel="Predicted concentration (mM)",
           title="Grouped cross-validation parity")
    fig.tight_layout(); fig.savefig(outdir / "prediction_parity.png", dpi=220); plt.close(fig)

    error = best.predicted_mM - best.actual_mM
    mean = (best.predicted_mM + best.actual_mM) / 2
    bias, sd = error.mean(), error.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(mean, error, alpha=.7)
    for value, style in [(bias, "-"), (bias-1.96*sd, "--"), (bias+1.96*sd, "--")]:
        ax.axhline(value, linestyle=style, color="black", linewidth=1)
    ax.set(xlabel="Mean of reference and prediction (mM)", ylabel="Prediction error (mM)",
           title="Bland–Altman agreement")
    fig.tight_layout(); fig.savefig(outdir / "bland_altman.png", dpi=220); plt.close(fig)
