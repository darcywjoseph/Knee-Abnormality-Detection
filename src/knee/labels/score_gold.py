"""Score extracted label probabilities against the gold set."""
from __future__ import annotations

import glob
import json
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.common import TARGETS

GOLD_CSV = "data/weak_labels/gold_labels.csv"


def load_predictions(pred_glob: str) -> pd.DataFrame:
    """Loads every prediction chunk matching a glob into one table.

    Args:
        pred_glob: Shell glob over JSON arrays of {"uid", <label>: probability}.

    Returns:
        The predictions indexed by study UID, with the twelve TARGETS columns.

    Raises:
        ValueError: No file matches, or a chunk is missing one of the twelve labels.
    """
    paths = sorted(glob.glob(pred_glob))
    if not paths:
        raise ValueError(f"no prediction files match {pred_glob!r}")
    rows: list[dict] = []
    for path in paths:
        with open(path) as f:
            rows += json.load(f)
    pred = pd.DataFrame(rows).set_index("uid")
    missing = [t for t in TARGETS if t not in pred.columns]
    if missing:
        raise ValueError(f"{pred_glob}: predictions are missing labels {missing}")
    return pred[TARGETS]


def score(pred_glob: str, gold_csv: str = GOLD_CSV) -> pd.DataFrame:
    """Scores one extraction run against the gold labels, label by label.

    Args:
        pred_glob: Shell glob over the prediction chunks, as for `load_predictions`.
        gold_csv: CSV with StudyInstanceUID and the twelve labels as 0/1.

    Returns:
        A table indexed by label with the positive count on the scored studies and the
        ROC AUC. A label with only one class present is dropped, so the frame can be
        shorter than twelve rows.

    Raises:
        ValueError: The predictions cover none of the gold studies.
    """
    pred = load_predictions(pred_glob)
    gold = pd.read_csv(gold_csv).set_index("StudyInstanceUID")
    shared = gold.index.intersection(pred.index)
    if not len(shared):
        raise ValueError(f"{pred_glob}: no gold study appears in the predictions")
    if len(shared) < len(gold):
        print(f"note: {len(shared)} of {len(gold)} gold studies are covered by {pred_glob}")
    gold, pred = gold.loc[shared], pred.loc[shared]
    rows = []
    for label in TARGETS:
        y = gold[label].astype(float)
        if y.nunique() < 2:
            print(f"note: {label} has one class on these studies; skipped")
            continue
        rows.append((label, int(y.sum()), float(roc_auc_score(y, pred[label].astype(float)))))
    return pd.DataFrame(rows, columns=["label", "n_pos", "auc"]).set_index("label")


def main(pred_glob: str) -> None:
    """Prints the per-label AUC table and its mean.

    Args:
        pred_glob: Shell glob over the prediction chunks.
    """
    res = score(pred_glob)
    print(res.round(3).to_string())
    print(f"\nMEAN AUC: {res['auc'].mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m knee.labels.score_gold '<pred_glob>'")
    main(sys.argv[1])
