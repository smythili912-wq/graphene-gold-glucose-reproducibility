# Reproducible analysis for a graphene–gold glucose metasurface

This repository is a transparent analysis companion for the manuscript *Machine Learning-Enhanced Graphene-Gold Hybrid Metasurface Architecture for Precision Glucose Monitoring*. It validates raw spectral tables, prevents batch leakage, calculates calibration and uncertainty statistics, compares classical regressors with a one-dimensional CNN, and produces reviewer-facing reports.

> **Scientific boundary.** The bundled demo data are synthetic and exist only to verify the software. They are not experimental evidence and must not be used to support sensitivity, selectivity, detection-limit, non-invasive, biological-matrix, or clinical claims. Replace them with traceable instrument exports before reporting results.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -e .
python run_pipeline.py --demo
pytest -q
```

For real data, populate `data/experimental_spectra.csv` and `data/sample_metadata.csv`, set `demo_mode: false` in `config.yaml`, and run `python run_pipeline.py`. Outputs are written to `data/results/`.

## Required experimental design

- Preserve every raw spectrum; never replace replicates with averages.
- Use unique `sample_id`, `sensor_id`, `batch_id`, and `acquisition_id` values.
- Record blanks, independently prepared concentrations, technical replicates, sensor batches, acquisition order, matrix, temperature, flow rate, and instrument settings.
- Split by independent experimental group—not by individual spectrum. Augmented variants inherit their parent group and remain training-only.
- Keep an untouched external batch or acquisition campaign for final validation.
- Describe solution-only work as *in-vitro buffer sensing*, not non-invasive monitoring.

## Pipeline outputs

The pipeline creates validated data audits, calibration coefficients with bootstrap confidence intervals, blank-based LOD/LOQ, grouped cross-validation metrics, leave-one-batch-out results, residual and Bland–Altman diagnostics, robustness stress tests, and a machine-readable `metrics.json`. Every output records the random seed and input hashes.

## Repository structure

Only two source folders are used: `src/` contains analysis code and `data/` contains inputs, schemas, and generated results. Reviewer-facing documentation and the executable entry point remain at repository root.

## Claim policy

Software reproducibility cannot repair absent experiments. Biological-matrix validation, independent fabrication batches, raw spectra, blank replicates, and external validation must be genuinely collected. The pipeline fails closed when required provenance is missing and labels demonstration results prominently.
