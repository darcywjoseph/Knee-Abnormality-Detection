"""Compares two experiments pooled over the same folds on the three offline judges.

Ranks are taken within each fold before pooling, as the ensemble does, so five separately
trained models never mix their output scales. Each label is then scored against the weak
labels binarised at 0.5, against the confident rows only (|p - 0.5| > 0.4) and against the
expert-labelled gold studies. Paired bootstraps on gold and on the weak judge say how far the
mean delta sits from zero. This is the verdict tool for a five-fold sweep against its control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.common import TARGETS, load_weak, rank_normalise


def pooled(tag: str, folds: list[int]) -> pd.DataFrame:
    """Loads one experiment's out-of-fold predictions, rank-normalised within each fold.

    Args:
        tag: The experiment tag, as in data/oof/oof_fold<k>_<tag>.csv.
        folds: The folds to pool.

    Returns:
        The pooled frame indexed by StudyInstanceUID, label columns as within-fold ranks and
        the `source` column carried through.
    """
    parts = [
        rank_normalise(pd.read_csv(f"data/oof/oof_fold{f}_{tag}.csv").set_index("StudyInstanceUID"))
        for f in folds
    ]
    return pd.concat(parts)


def compare(new: str, ref: str, folds: list[int], n_gold: int = 2000, n_weak: int = 300,
            weak_csv: str | None = None) -> pd.DataFrame:
    """Scores two experiments on the three judges and prints the comparison.

    Args:
        new: Tag of the candidate experiment.
        ref: Tag of the control experiment.
        folds: Folds both experiments have OOF files for.
        n_gold: Bootstrap draws for the gold judge.
        n_weak: Bootstrap draws for the weak judge.
        weak_csv: Label file for the weak and confident-row judges, or None for the
            default v2 file. Gold rows are the same in every label file.

    Returns:
        A per-label table with the three judges for both experiments and their deltas.
    """
    weak = load_weak(Path(weak_csv)) if weak_csv else load_weak()
    a, b = pooled(new, folds), pooled(ref, folds)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    y = weak.loc[idx, TARGETS]
    yb = y.ge(0.5).astype(int)
    conf = (y - 0.5).abs() > 0.4
    gold = (a["source"] == "gold").to_numpy()
    yg = yb[gold]
    rows = []
    for t in TARGETS:
        r = dict(
            label=t,
            weak_n=roc_auc_score(yb[t], a[t]), weak_r=roc_auc_score(yb[t], b[t]),
            conf_n=roc_auc_score(yb.loc[conf[t], t], a.loc[conf[t], t]),
            conf_r=roc_auc_score(yb.loc[conf[t], t], b.loc[conf[t], t]),
        )
        if yg[t].nunique() > 1:
            r["gold_n"] = roc_auc_score(yg[t], a.loc[gold, t])
            r["gold_r"] = roc_auc_score(yg[t], b.loc[gold, t])
        rows.append(r)
    df = pd.DataFrame(rows).set_index("label")
    for k in ("weak", "conf", "gold"):
        df[f"d_{k}"] = df[f"{k}_n"] - df[f"{k}_r"]
    print(f"{new} vs {ref} | folds {folds} | {len(idx)} studies | {int(gold.sum())} gold")
    print(df.round(3).to_string())
    print(
        f"MEAN weak {df.weak_n.mean():.4f} vs {df.weak_r.mean():.4f} ({df.d_weak.mean():+.4f}) | "
        f"conf {df.conf_n.mean():.4f} vs {df.conf_r.mean():.4f} ({df.d_conf.mean():+.4f}) | "
        f"gold {df.gold_n.mean():.4f} vs {df.gold_r.mean():.4f} ({df.d_gold.mean():+.4f})"
    )
    rng = np.random.default_rng(0)
    Y, A, B = yg.to_numpy(), a.loc[gold, TARGETS].to_numpy(), b.loc[gold, TARGETS].to_numpy()
    d = []
    for _ in range(n_gold):
        rr = rng.integers(0, len(Y), len(Y))
        vals = [
            roc_auc_score(Y[rr, j], A[rr, j]) - roc_auc_score(Y[rr, j], B[rr, j])
            for j in range(len(TARGETS)) if Y[rr, j].min() != Y[rr, j].max()
        ]
        if vals:
            d.append(np.mean(vals))
    d = np.array(d)
    print(
        f"paired bootstrap gold: {d.mean():+.4f} [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}] "
        f"P(>0)={np.mean(d > 0):.2f}"
    )
    Yw, Aw, Bw = yb.to_numpy(), a[TARGETS].to_numpy(), b[TARGETS].to_numpy()
    dw = []
    for _ in range(n_weak):
        rr = rng.integers(0, len(Yw), len(Yw))
        dw.append(np.mean([
            roc_auc_score(Yw[rr, j], Aw[rr, j]) - roc_auc_score(Yw[rr, j], Bw[rr, j])
            for j in range(len(TARGETS))
        ]))
    dw = np.array(dw)
    print(
        f"paired bootstrap weak: {dw.mean():+.4f} [{np.percentile(dw, 5):+.4f}, {np.percentile(dw, 95):+.4f}] "
        f"P(>0)={np.mean(dw > 0):.2f}"
    )
    return df


def main(argv: list[str]) -> int:
    """Runs the comparison from the command line.

    Args:
        argv: The candidate tag, the control tag, then the folds to pool; `--weak <file>`
            anywhere in the list scores the weak judges on that label file.

    Returns:
        The process exit status.
    """
    weak_csv = None
    if "--weak" in argv:
        i = argv.index("--weak")
        weak_csv = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if len(argv) < 3:
        print("arguments: <new tag> <ref tag> <fold> [<fold> ...] [--weak <label csv>]",
              file=sys.stderr)
        return 2
    compare(argv[0], argv[1], [int(f) for f in argv[2:]], weak_csv=weak_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
