# MultiOmics Alterations in HUUC

Statistical analysis of mother–newborn lipidome and metabolome cohorts (Kenya
and Zambia), comparing HIV‑exposed and HIV‑unexposed pregnancies and newborns.
The pipeline runs a regularized logistic-regression model with stability
selection to identify the biomarkers most consistently associated with HIV
exposure, and renders the corresponding box plots, stability/lasso paths, and
volcano plots from cached results.

## Top-level directories

| Path | Purpose |
| --- | --- |
| `configs/` | YAML-driven configuration (cohorts, file mappings, marker name maps). |
| `common/` | Shared library: data loading, feature engineering, plotting / modeling utilities. |
| `maternal_HIV_effects/` | Maternal-side analysis (load, describe, stability-select, lasso-path, volcano). |
| `neonatal_HIV_exposure_biomarkers/` | Neonatal-side analysis (HEEL-prick data, stability path, volcano, violin). |
| `results/` | Generated PCA / PLSR / ROC / volcano / UMAP figures. |
| `script.py` | CLI entry point that re-renders figures from cached selection results. |
| `utils.py` | Verbose-printing helper used by the data loaders. |
| `requirements.txt` | Python dependencies. |
