# Reproducibility protocol

## Experimental hierarchy

The inferential unit is an independently prepared sample measured on an independently identified sensor. Technical spectra are nested within samples; samples are nested within fabrication batches. Confidence intervals and model splits must respect this hierarchy.

Minimum recommended evidence is at least three independently fabricated batches, independently prepared concentration levels spanning blanks and the claimed analytical range, and at least three independent samples per level and batch. This is a design recommendation, not permission to invent observations.

## Statistical analysis

Calibration reports slope, intercept, residual diagnostics, coefficient of determination, RMSE, MAE, and cluster-aware bootstrap 95% confidence intervals. LOD is calculated from observed blank variability and the local calibration slope (`3σ_blank/S`); LOQ uses `10σ_blank/S`. The report includes the number of independent blanks and warns when the estimate is underpowered.

CNN and baseline models use grouped folds. Any preprocessing learned from data is fitted inside each training fold. Augmentation is restricted to training data. Final performance must be reported on an untouched external experimental group, with uncertainty intervals and per-matrix results.

## Reproducibility controls

Random seeds, package versions, configuration, input SHA-256 hashes, exclusions, and validation warnings are saved with every run. Duplicate identifiers, inconsistent labels, non-monotonic wavelength axes, missing provenance, and cross-split group overlap stop execution.
