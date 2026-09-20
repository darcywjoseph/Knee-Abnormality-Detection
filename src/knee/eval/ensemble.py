"""Pool several experiments' per-fold OOF into one rank-mean ensemble and score it.

The metric depends only on the ordering within each label, so averaging ranks is lossless for
it and immune to two models disagreeing about absolute scale, which they do: nothing in the
training objective calibrates them against each other. Ranks are taken within each fold before
pooling, so five separately trained models' output scales never mix into one ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knee.common import TARGETS, gold_auc, load_oof, load_weak, paired_bootstrap, weak_auc

DEFAULT_PATTERNS = ["data/oof/oof_fold*_exp015.csv", "data/oof/oof_fold*_exp019.csv"]


def member_name(pattern: str) -> str:
    """Names an ensemble member after the distinguishing part of its file pattern.

    'data/oof/oof_fold*_exp015.csv' becomes 'exp015'.

    Args:
        pattern: The glob the member's OOF files were loaded from.

    Returns:
        A short label for the member, falling back to the pattern's stem.
    """
    stem = Path(pattern).stem
    _, _, rest = stem.partition("*")
    return rest.lstrip("_") or stem


def score(df: pd.DataFrame, weak: pd.DataFrame, label: str) -> tuple[float, float]:
    """Prints and returns one prediction set's gold and weak mean AUCs.

    Args:
        df: Predictions indexed by StudyInstanceUID, with a `source` column.
        weak: Table from `load_weak`.
        label: Row label for the printed line.

    Returns:
        The mean gold AUC and the mean weak AUC, each over the scorable labels.
    """
    mean_gold, _ = gold_auc(df, weak)
    mean_weak, _ = weak_auc(df, weak)
    n_gold = int((df["source"] == "gold").sum())
    print(f"{label:28s} gold {mean_gold:.4f} ({n_gold} studies) | weak {mean_weak:.4f} ({len(df)})")
    return mean_gold, mean_weak


def run(patterns: list[str], n_boot: int = 2000) -> None:
    """Scores each member, their rank-mean ensemble, and the paired gold-AUC difference.

    Args:
        patterns: One glob per member, each over that experiment's per-fold OOF files.
        n_boot: Resamples for the paired bootstrap.

    Raises:
        ValueError: A pattern matches no file, or a member has duplicate studies.
    """
    weak = load_weak()
    members = [load_oof(p) for p in patterns]
    shared = members[0].index
    for m in members[1:]:
        shared = shared.intersection(m.index)
    members = [m.loc[shared] for m in members]

    print(f"{len(members)} members | {len(shared)} studies\n")
    singles = [score(m, weak, f"member {i}: {member_name(p)}")
               for i, (m, p) in enumerate(zip(members, patterns))]

    ens = members[0].copy()
    ens[TARGETS] = np.mean([m[TARGETS].to_numpy() for m in members], axis=0)
    print()
    ens_gold, ens_weak = score(ens, weak, "RANK-MEAN ENSEMBLE")
    print(f"\ngain over the best single member: "
          f"gold {ens_gold - max(s[0] for s in singles):+.4f} | "
          f"weak {ens_weak - max(s[1] for s in singles):+.4f}")

    best = int(np.argmax([s[0] for s in singles]))
    mean_d, lo, hi, p_gain, n_used = paired_bootstrap(ens, members[best], weak, n_boot=n_boot)
    n_gold = int((ens["source"] == "gold").sum())
    print(f"paired bootstrap on gold (n={n_gold}) vs {member_name(patterns[best])}: "
          f"{mean_d:+.4f}  90% CI [{lo:+.4f}, {hi:+.4f}]  P(>0)={p_gain:.3f}")
    print(f"bootstrap draws used: {n_used}/{n_boot} ({n_boot - n_used} dropped as degenerate)")


def main(argv: list[str]) -> int:
    """Runs the ensemble report over the given patterns, or the default pair.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        A process exit status: 0 on success, 2 on a data error.
    """
    try:
        run(argv or DEFAULT_PATTERNS)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
