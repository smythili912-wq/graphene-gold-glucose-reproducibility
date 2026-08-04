# Data contract

`experimental_spectra.csv` must be a long table with one row per wavelength point: `sample_id,wavelength_nm,reflectance`. `sample_metadata.csv` contains one row per independent sample and must include `sample_id,sensor_id,batch_id,acquisition_id,matrix,concentration_mM,replicate_type,is_synthetic`.

Keep raw values unchanged. Corrections and normalization are performed by code. Never copy augmented spectra into the raw table. Biological-matrix measurements should also record donor/sample provenance, matrix preparation, reference-assay value, dilution, recovery, pH, temperature, and ethics status where applicable.

The CSV files shipped here are schema templates only. `run_pipeline.py --demo` generates labelled synthetic data inside `data/results/`; it does not alter the templates.
