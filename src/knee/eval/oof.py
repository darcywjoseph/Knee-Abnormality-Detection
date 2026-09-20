"""Score out-of-fold predictions: pooled across folds, or one fold under three judges.

Every study is predicted by the one model that never saw it, so concatenating the folds gives
out-of-fold predictions for every study, gold ones included. The pooled gold AUC is the closest
thing to a leaderboard read.
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.common import (
    TARGETS,
    bootstrap_gold_ci,
    gold_auc,
    load_oof,
    load_weak,
    per_label_auc,
    weak_auc,
)

CONF_MARGIN = 0.4
JUDGES = ("weak", "conf", "gold")


def print_skipped(per: dict[str, float]) -> None:
    """Prints the labels a judge could not score, if any.

    Args:
        per: Label to AUC, as returned by `per_label_auc`.
    """
    missing = [t for t in TARGETS if t not in per]
    if missing:
        print(f"  (skipped, one class only: {', '.join(missing)})")


def pooled_report(pattern: str, n_boot: int = 2000) -> None:
    """Prints the pooled gold AUC, its bootstrap interval, and the pooled weak AUC.

    Args:
        pattern: Glob over one experiment's per-fold OOF files.
        n_boot: Bootstrap resamples for the gold interval.

    Raises:
        ValueError: No file matches, or a study appears in more than one fold.
    """
    oof = load_oof(pattern)
    weak = load_weak()
    n_gold = int((oof["source"] == "gold").sum())
    print(f"{len(oof)} studies | {n_gold} gold")

    mean_gold, per = gold_auc(oof, weak)
    lo, hi, n_used = bootstrap_gold_ci(oof, weak, n_boot=n_boot)

    print("\npooled GOLD AUC, all gold studies")
    for label, value in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"  {label:18s} {value:.4f}")
    print_skipped(per)
    print(f"\nMEAN over {len(per)}/12 labels = {mean_gold:.4f}   90% CI [{lo:.3f}, {hi:.3f}]")
    print(f"bootstrap draws used: {n_used}/{n_boot}")

    mean_weak, weak_per = weak_auc(oof, weak)
    print(f"pooled weak AUC (all {len(oof)} studies, {len(weak_per)}/12 labels) = "
          f"{mean_weak:.4f}")
    print_skipped(weak_per)


def fold_judges(pred: pd.DataFrame, weak: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Scores one fold's predictions under three judges.

    The judges are, in increasing order of trust: `weak`, the AUC against every LLM weak label,
    which rewards agreeing with the extractor; `conf`, the AUC over the rows the report asserted
    or denied outright; and `gold`, the AUC on this fold's radiologist-labelled studies, noisy
    because there are few of them but the only judge with no LLM in the loop.

    Args:
        pred: One fold's OOF predictions indexed by StudyInstanceUID, with a `source` column.
        weak: Table from `load_weak`.

    Returns:
        Judge name to a label-to-AUC mapping. Unscorable labels are absent.
    """
    y = weak.loc[pred.index]
    y_bin = y[TARGETS].ge(0.5).astype(int)
    confident = (y[TARGETS] - 0.5).abs() >= CONF_MARGIN

    conf = {}
    for label in TARGETS:
        rows = confident[label]
        if y_bin.loc[rows, label].nunique() > 1:
            conf[label] = roc_auc_score(y_bin.loc[rows, label], pred.loc[rows, label])

    return {"weak": per_label_auc(y_bin, pred[TARGETS]),
            "conf": conf,
            "gold": gold_auc(pred, weak)[1]}


def fold_report(runs: dict[str, str]) -> None:
    """Prints one table per judge comparing the named single-fold OOF files.

    Args:
        runs: Run name to the path of that run's single-fold OOF CSV.

    Raises:
        ValueError: A path matches no file, or holds duplicate studies.
    """
    weak = load_weak()
    preds = {name: load_oof(path, rank_within_fold=False) for name, path in runs.items()}
    results = {name: fold_judges(p, weak) for name, p in preds.items()}
    n_gold = int((next(iter(preds.values()))["source"] == "gold").sum())

    for judge in JUDGES:
        table = pd.DataFrame({name: results[name][judge] for name in runs}).reindex(TARGETS)
        table.loc["MEAN"] = table.mean()
        suffix = f" (n={n_gold})" if judge == "gold" else ""
        print(f"\n== {judge} AUC, fold 0{suffix}")
        print(table.round(4).to_string())
        if len(runs) > 1:
            names = list(runs)
            delta = table.loc["MEAN", names[-1]] - table.loc["MEAN", names[-2]]
            print(f"delta {names[-1]} - {names[-2]}: {delta:+.4f}")


def main(argv: list[str]) -> int:
    """Runs the pooled report, or the per-fold judge tables under `--fold`.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        A process exit status: 0 on success, 2 on a usage or data error.
    """
    args = [a for a in argv if a != "--fold"]
    try:
        if "--fold" in argv:
            if not args:
                raise ValueError("--fold needs at least one name=path argument")
            fold_report(dict(a.split("=", 1) for a in args))
        else:
            pooled_report(args[0] if args else "data/oof/oof_fold*.csv")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
