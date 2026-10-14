# Knee-Abnormality-Detection

A classifier for the Kaggle RSNA Knee Abnormality Detection competition:
twelve knee findings from MRI, scored by mean ROC AUC.

## Setup

```bash
uv sync --extra dev --extra kaggle
```

## Layout

- `src/knee/`: the package, by stage. `cache.py` (DICOM to a six-slot slice cache),
  `dataset.py`, `model.py`, `train.py`, `predict.py`; `data/` (folds, slice directions);
  `labels/` (gold scoring, chunk validation); `eval/` (single-fold scoring, rank-mean
  ensembles, two-experiment judge, attention diagnostics, figures); `infer/dryrun.py`
  (runs a submission kernel's cells on a synthetic cache).
- `scripts/`: thin entry points (`build_cache.py`, `train.py`, `predict.py`).
- `notebooks/`: `02_train`, `03_cache`, `04_predict` import the package;
  `02_baseline_train_2p5d`, `03_cache_builder_v3`, `03_cache_builder_v4` and `submissions/`
  are the standalone Kaggle kernels, maintained by hand.
- `docs/`, `experiments/`, `data/`, `local/`: private working material (state, experiment
  ledger, labels, folds, out-of-fold predictions, cloud and Kaggle tooling). Not in this
  repository.

## Running

```bash
uv run python scripts/train.py --overrides '{"FOLD":0,"OUT_DIR":"runs/example_f0"}'
uv run python -m knee.eval.oof --fold name=runs/example_f0/oof_fold0.csv
uv run python -m knee.eval.ensemble 'data/oof/oof_fold*_expA.csv' 'data/oof/oof_fold*_expB.csv'
uv run python -m knee.eval.judge expA expB 0 1 2 3 4
uv run python -m knee.infer.dryrun --kernel notebooks/submissions/submition-02.ipynb
```
