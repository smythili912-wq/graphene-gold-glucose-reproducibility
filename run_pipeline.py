#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.calibration import cluster_bootstrap, fit_calibration
from src.data_validation import sha256, validate_inputs
from src.evaluation import grouped_evaluate
from src.figures import save_figures
from src.preprocessing import matrix_from_long
from src.reporting import write_report
from src.robustness import stress_test
from src.spectral_analysis import add_shift, resonance_features


ROOT = Path(__file__).resolve().parent


def synthetic_demo(seed: int):
    """Create clearly labelled software-test spectra, never experimental evidence."""
    rng = np.random.default_rng(seed)
    wavelengths = np.arange(640.0, 701.0, 0.5)
    spectra, metadata = [], []
    concentrations = [0, 0, 0.1, 0.5, 1, 2, 5, 10]
    for batch in range(1, 6):
        batch_offset = rng.normal(0, 0.18)
        for rep, concentration in enumerate(concentrations, 1):
            sample = f"DEMO-B{batch:02d}-S{rep:02d}"
            center = 670 + batch_offset + 1.8 * np.log1p(concentration)
            reflectance = 0.78 - 0.22 * np.exp(-0.5*((wavelengths-center)/2.8)**2)
            reflectance += rng.normal(0, 0.0025, wavelengths.size)
            spectra.extend((sample, float(w), float(r)) for w, r in zip(wavelengths, reflectance))
            metadata.append((sample, f"DEMO-SENSOR-{batch:02d}", f"DEMO-BATCH-{batch:02d}",
                             f"DEMO-ACQ-{batch:02d}-{rep:02d}", "synthetic_demo",
                             concentration, "independent_demo_sample", True))
    return (pd.DataFrame(spectra, columns=["sample_id", "wavelength_nm", "reflectance"]),
            pd.DataFrame(metadata, columns=["sample_id", "sensor_id", "batch_id",
                                           "acquisition_id", "matrix", "concentration_mM",
                                           "replicate_type", "is_synthetic"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true", help="Run only with labelled synthetic data")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    seed = int(cfg["project"]["seed"])
    demo = args.demo or bool(cfg["project"]["demo_mode"])
    if demo:
        spectra, metadata = synthetic_demo(seed)
    else:
        spectra = pd.read_csv(ROOT / cfg["data"]["spectra"])
        metadata = pd.read_csv(ROOT / cfg["data"]["metadata"])

    warnings = validate_inputs(spectra, metadata, demo)
    out = ROOT / cfg["data"]["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    spectra.to_csv(out / "analysis_spectra.csv", index=False)
    metadata.to_csv(out / "analysis_metadata.csv", index=False)

    prep = cfg["preprocessing"]
    X, ids, axis = matrix_from_long(spectra, {"degree": prep["baseline_degree"],
                                              "window": prep["savgol_window"],
                                              "order": prep["savgol_order"]})
    indexed = metadata.set_index("sample_id").loc[ids]
    y = indexed.concentration_mM.to_numpy(float)
    groups = indexed[cfg["validation"]["group_column"]].astype(str).to_numpy()
    features = add_shift(resonance_features(spectra), metadata)
    features.to_csv(out / "spectral_features.csv", index=False)

    cal = fit_calibration(features)
    cal.update(cluster_bootstrap(features, cfg["validation"]["bootstrap_iterations"],
                                 cfg["validation"]["alpha"], seed))
    pd.DataFrame([cal]).to_csv(out / "calibration.csv", index=False)
    cv, predictions = grouped_evaluate(X, y, groups, cfg["validation"]["folds"], seed)
    cv.to_csv(out / "grouped_cv_metrics.csv", index=False)
    predictions.to_csv(out / "grouped_cv_predictions.csv", index=False)
    robust = stress_test(X, y, cfg["robustness"], seed)
    robust.to_csv(out / "robustness.csv", index=False)
    save_figures(features, predictions, out)

    payload = {"status": "DEMONSTRATION_ONLY" if demo else "EXPERIMENTAL_ANALYSIS",
               "seed": seed, "warnings": warnings, "calibration": cal,
               "grouped_cv_mean": cv.groupby("model")[["rmse_mM", "mae_mM", "r2"]].mean().to_dict("index"),
               "input_hashes": {"config": sha256(cfg_path)},
               "scientific_boundary": "Synthetic results cannot support manuscript claims." if demo else
                                      "Claims remain limited to the supplied matrices and design."}
    write_report(out / "metrics.json", payload)
    (out / "RUN_STATUS.txt").write_text(payload["status"] + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "results": str(out), "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
