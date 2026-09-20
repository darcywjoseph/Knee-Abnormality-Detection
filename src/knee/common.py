"""Constants and scoring shared by every analysis script in this repo.

The competition metric is the unweighted mean of twelve per-label ROC AUCs. It depends only on
the ordering within each label, so predictions from separately trained models are combined by
rank, never by probability. Everything that scores an OOF file goes through this module, so the
metric cannot drift between scripts.

The Kaggle notebooks keep their own copies of these constants because a kernel cannot import
from the repo.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

__all__ = [
    "TARGETS", "SLOT_NAMES", "CACHE_GLOB", "WEAK_CSV", "load_weak", "load_oof",
    "rank_normalise", "per_label_auc", "gold_auc", "weak_auc", "bootstrap_gold_ci",
    "paired_bootstrap", "slice_descriptors",
]

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Cache v3 slot order.
SLOT_NAMES = ["sagittal FS", "coronal FS", "axial FS", "sagittal non-FS", "coronal T1",
              "sagittal T1"]
CACHE_GLOB = "data/cache_v3/s*/cache/*.npz"
WEAK_CSV = Path("data/weak_labels/train_weak_labels.csv")


def load_weak(path: Path = WEAK_CSV) -> pd.DataFrame:
    """Loads the weak-label table indexed by study.

    Args:
        path: CSV with StudyInstanceUID, the twelve label columns as probabilities in [0, 1],
            and a `source` column. Gold studies carry their radiologist labels here as 0/1.

    Returns:
        The table indexed by StudyInstanceUID.
    """
    return pd.read_csv(path).set_index("StudyInstanceUID")


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces each label column by its rank in (0, 1), keeping the other columns.

    Args:
        df: Predictions with the TARGETS columns, one row per study.

    Returns:
        A copy where each TARGETS column is rankdata / (n + 1).
    """
    out = df.copy()
    n = len(df) + 1.0
    for t in TARGETS:
        out[t] = rankdata(df[t].to_numpy()) / n
    return out


def load_oof(pattern: str, rank_within_fold: bool = True) -> pd.DataFrame:
    """Concatenates one experiment's per-fold OOF files into one table.

    Args:
        pattern: Glob such as 'data/oof/oof_fold*_exp019.csv'.
        rank_within_fold: Rank-normalise each fold before concatenating. Five separately
            trained models share no output scale, so pooling raw probabilities mixes five
            orderings into one; ranking within a fold first keeps them comparable.

    Returns:
        All folds indexed by StudyInstanceUID, with the `source` column kept.

    Raises:
        ValueError: No file matches, or a study appears in more than one fold.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"no OOF files match {pattern!r}")
    parts = []
    for f in files:
        fold = pd.read_csv(f).set_index("StudyInstanceUID")
        parts.append(rank_normalise(fold) if rank_within_fold else fold)
    out = pd.concat(parts)
    if out.index.duplicated().any():
        raise ValueError(f"{pattern}: a study appears in more than one fold")
    return out


def per_label_auc(y: pd.DataFrame, p: pd.DataFrame) -> dict[str, float]:
    """Computes ROC AUC per label, skipping labels with one class.

    Args:
        y: Binary truth with the TARGETS columns.
        p: Scores with the TARGETS columns, same index as `y`.

    Returns:
        Label to AUC for every label with both classes present.
    """
    return {t: roc_auc_score(y[t], p[t]) for t in TARGETS if y[t].nunique() > 1}


def gold_auc(pred: pd.DataFrame, weak: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Scores predictions on the radiologist-labelled studies only.

    Args:
        pred: Predictions with a `source` column; rows with source == 'gold' are scored.
        weak: Table from `load_weak`; gold rows hold the radiologist 0/1 labels.

    Returns:
        The mean AUC over scorable labels and the per-label AUCs.
    """
    gold = pred[pred["source"] == "gold"]
    per = per_label_auc(weak.loc[gold.index, TARGETS].ge(0.5).astype(int), gold[TARGETS])
    return float(np.mean(list(per.values()))), per


def weak_auc(pred: pd.DataFrame, weak: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Scores predictions on every study against the LLM weak labels.

    This judge rewards agreeing with the label extractor, so it is a stability check rather
    than a measure of clinical accuracy.

    Args:
        pred: Predictions with the TARGETS columns, indexed by StudyInstanceUID.
        weak: Table from `load_weak`.

    Returns:
        The mean AUC over scorable labels and the per-label AUCs.
    """
    per = per_label_auc(weak.loc[pred.index, TARGETS].ge(0.5).astype(int), pred[TARGETS])
    return float(np.mean(list(per.values()))), per


def _mean_auc(y: np.ndarray, p: np.ndarray, rows: np.ndarray) -> float | None:
    """Means the per-label AUCs of one bootstrap resample.

    Args:
        y: Binary truth of shape (studies, labels).
        p: Scores of the same shape.
        rows: Row indices of the resample.

    Returns:
        The mean AUC over the labels that still carry both classes, or None when no label
        does and the draw is unusable.
    """
    aucs = [roc_auc_score(y[rows, j], p[rows, j])
            for j in range(y.shape[1]) if y[rows, j].min() != y[rows, j].max()]
    return float(np.mean(aucs)) if aucs else None


def bootstrap_gold_ci(pred: pd.DataFrame, weak: pd.DataFrame, n_boot: int = 2000,
                      seed: int = 0) -> tuple[float, float, int]:
    """Bootstraps the 90% interval of the mean gold AUC by resampling studies.

    Args:
        pred: As for `gold_auc`.
        weak: As for `gold_auc`.
        n_boot: Resamples to draw.
        seed: RNG seed.

    Returns:
        The 5th percentile, the 95th percentile, and how many draws were usable. A draw is
        unusable when every label has one class; the caller should report that count.
    """
    gold = pred[pred["source"] == "gold"]
    y = weak.loc[gold.index, TARGETS].ge(0.5).astype(int).to_numpy()
    p = gold[TARGETS].to_numpy()
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        mean = _mean_auc(y, p, rng.integers(0, len(gold), len(gold)))
        if mean is not None:
            means.append(mean)
    lo, hi = np.percentile(means, [5, 95])
    return float(lo), float(hi), len(means)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, weak: pd.DataFrame, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float, float, float, int]:
    """Bootstraps the gold-AUC difference between two prediction sets on the same studies.

    Both sets score the same resampled studies each draw, so the interval is on the paired
    difference, which is much tighter than two separate intervals.

    Args:
        a: Predictions with a `source` column, e.g. an ensemble.
        b: Predictions on the same index, e.g. its best member.
        weak: As for `gold_auc`.
        n_boot: Resamples to draw.
        seed: RNG seed.

    Returns:
        Mean difference a minus b, its 5th and 95th percentiles, P(difference > 0), and how
        many draws were usable.
    """
    gold_idx = a.index[a["source"] == "gold"]
    y = weak.loc[gold_idx, TARGETS].ge(0.5).astype(int).to_numpy()
    pa = a.loc[gold_idx, TARGETS].to_numpy()
    pb = b.loc[gold_idx, TARGETS].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(gold_idx), len(gold_idx))
        mean_a = _mean_auc(y, pa, rows)
        if mean_a is not None:
            diffs.append(mean_a - _mean_auc(y, pb, rows))
    d = np.array(diffs)
    return (float(d.mean()), float(np.percentile(d, 5)), float(np.percentile(d, 95)),
            float(np.mean(d > 0)), len(d))


def slice_descriptors(vol: np.ndarray) -> np.ndarray:
    """Reduces each slice to a coarse, mean-removed, unit-norm vector.

    A dot product between two descriptors is then a correlation, which is what the
    slice-direction test needs.

    Args:
        vol: Slices of one slot, shape (S, H, W); H and W must be multiples of 20.

    Returns:
        Descriptors of shape (S, 400).
    """
    s, h, w = vol.shape
    x = vol.reshape(s, 20, h // 20, 20, w // 20).mean(axis=(2, 4)).reshape(s, -1)
    x = x.astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)
