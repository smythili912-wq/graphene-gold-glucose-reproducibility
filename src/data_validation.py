from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SPECTRAL_COLUMNS = {"sample_id", "wavelength_nm", "reflectance"}
META_COLUMNS = {
    "sample_id", "sensor_id", "batch_id", "acquisition_id", "matrix",
    "concentration_mM", "replicate_type", "is_synthetic"
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_inputs(spectra: pd.DataFrame, metadata: pd.DataFrame, demo_mode: bool) -> list[str]:
    missing_s = SPECTRAL_COLUMNS - set(spectra)
    missing_m = META_COLUMNS - set(metadata)
    if missing_s or missing_m:
        raise ValueError(f"Missing columns: spectra={sorted(missing_s)}, metadata={sorted(missing_m)}")
    if metadata.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique in metadata")
    if spectra.duplicated(["sample_id", "wavelength_nm"]).any():
        raise ValueError("Duplicate sample/wavelength pairs detected")
    unknown = set(spectra.sample_id) - set(metadata.sample_id)
    if unknown:
        raise ValueError(f"Spectra contain sample IDs absent from metadata: {sorted(unknown)[:5]}")
    if not spectra.reflectance.between(0, 1.5).all():
        raise ValueError("Reflectance values must be between 0 and 1.5")
    if (metadata.concentration_mM < 0).any():
        raise ValueError("Concentration cannot be negative")
    for sample_id, group in spectra.groupby("sample_id", sort=False):
        if not group.wavelength_nm.is_monotonic_increasing:
            raise ValueError(f"Wavelength axis is not increasing for {sample_id}")
    warnings: list[str] = []
    synthetic = metadata.is_synthetic.astype(str).str.lower().isin({"true", "1", "yes"})
    if synthetic.any() and not demo_mode:
        raise ValueError("Synthetic rows are forbidden when demo_mode is false")
    if metadata.batch_id.nunique() < 3:
        warnings.append("Fewer than three independent batches; uncertainty may be unstable.")
    if metadata.query("concentration_mM == 0").shape[0] < 6:
        warnings.append("Fewer than six independent blank samples; LOD precision is weak.")
    if set(metadata.matrix.str.lower()) <= {"buffer", "water", "synthetic_demo"}:
        warnings.append("No biological matrix is present; non-invasive or clinical claims are unsupported.")
    return warnings


def assert_group_disjoint(train_groups, test_groups) -> None:
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {sorted(overlap)}")
