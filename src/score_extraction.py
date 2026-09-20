"""Score extracted label probabilities against the 58-study gold set.

Usage: .venv/bin/python src/score_extraction.py <pred_glob>
Each matched file: JSON array of {"uid", <12 label keys>}.
"""
from __future__ import annotations

import glob
import json
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

SCRATCH = (
    "/private/tmp/claude-501/-Users-darcyjoseph-personal-code-Knee-Abnormality-Detection/"
    "ca64d759-dc26-43c7-868b-6fd20ba6824d/scratchpad"
)


def score(pred_glob: str) -> pd.DataFrame:
    preds: list[dict] = []
    for path in sorted(glob.glob(pred_glob)):
        with open(path) as f:
            preds += json.load(f)
    pred = pd.DataFrame(preds).set_index("uid")
    gold = pd.read_csv(f"{SCRATCH}/gold_labels.csv").set_index("StudyInstanceUID")
    labels = list(gold.columns)
    pred = pred.loc[gold.index, labels]
    rows = [
        (c, int(gold[c].sum()), roc_auc_score(gold[c].values, pred[c].values))
        for c in labels
    ]
    return pd.DataFrame(rows, columns=["label", "n_pos", "auc"]).set_index("label")


if __name__ == "__main__":
    res = score(sys.argv[1])
    print(res.round(3).to_string())
    print(f"\nMEAN AUC: {res['auc'].mean():.4f}")
