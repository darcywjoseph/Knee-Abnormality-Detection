"""Build grouped, stratified cross-validation folds.

Groups are normalised-report hashes (duplicate reports are likely the same exam re-read;
the data carries no PatientID). Folds are balanced on report language and weak-label
prevalence.
"""
from __future__ import annotations

import hashlib
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

N_FOLDS = 5
SEED = 42
TRAIN_CSV = "data/train.csv"
WEAK_CSV = "data/weak_labels/train_weak_labels.csv"
OUT_CSV = "data/folds.csv"

LANG_MARKERS = {
    "tr": [" ve ", "izlenmedi", "saptanmadi", "düzey", "eklem", "sıvı", "menisküs"],
    "es": [" el ", " la ", " de la ", "sin ", "derrame", "rodilla", "menisco"],
    "nl": [" het ", " de ", "geen ", "knie", "gewricht", "vocht"],
    "de": [" der ", " und ", "kein", "erguss", "knie", "gelenk"],
    "fr": [" le ", " du ", "pas de", "genou", "épanchement", "ménisque"],
    "hr": ["koljen", "nema ", "zglob", "izljev", "menisk"],
}


def detect_lang(text: str) -> str:
    """Guesses the report language from alphabet and marker words.

    Args:
        text: One radiology report.

    Returns:
        A language code from LANG_MARKERS, 'el' or 'cyr' for a non-Latin alphabet, or
        'en' when no language scores at least two markers.
    """
    t = " " + text.lower() + " "
    if re.search(r"[Ͱ-Ͽ]", t):
        return "el"
    if re.search(r"[Ѐ-ӿ]", t):
        return "cyr"
    scores = {lang: sum(m in t for m in ms) for lang, ms in LANG_MARKERS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "en"


def normalize(text: str) -> str:
    """Normalises a report to NFKC lower case with runs of whitespace collapsed.

    Args:
        text: One radiology report, any type; it is coerced to str.

    Returns:
        The normalised text, used as the grouping key's input.
    """
    t = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", t).strip()


def assign_folds(groups: pd.DataFrame, labels: list[str]) -> dict[str, int]:
    """Assigns each report group to a fold, balancing size, language and prevalence.

    Candidate folds are those with the fewest studies of this group's language (which keeps
    language and size balanced); among the candidates the fold whose current label
    prevalence is furthest below the group's labels wins.

    Args:
        groups: One row per group, already shuffled, with columns `n`, `lang` and the
            label means.
        labels: The twelve weak-label column names.

    Returns:
        Group id to fold index.
    """
    fold_n = np.zeros(N_FOLDS)
    fold_lang = {lang: np.zeros(N_FOLDS) for lang in groups.lang.unique()}
    fold_lab = np.zeros((N_FOLDS, len(labels)))
    assignment = {}
    for gid, row in groups.iterrows():
        lc = fold_lang[row.lang]
        cands = np.flatnonzero(lc == lc.min())
        x = row[labels].values.astype(float)
        prev = fold_lab[cands] / np.maximum(fold_n[cands], 1)[:, None]
        k = cands[int(np.argmin(prev @ x))]
        assignment[gid] = int(k)
        fold_n[k] += row.n
        fold_lang[row.lang][k] += row.n
        fold_lab[k] += x * row.n
    return assignment


def main(out_csv: str = OUT_CSV) -> None:
    """Builds the folds and writes them, printing the balance tables.

    Args:
        out_csv: Where to write the (StudyInstanceUID, fold) table.

    Raises:
        ValueError: A report group was split across folds, which would leak between train
            and validation.
    """
    train = pd.read_csv(TRAIN_CSV)
    weak = pd.read_csv(WEAK_CSV)
    labels = weak.columns[1:13].tolist()
    df = train[["StudyInstanceUID", "Report"]].merge(weak, on="StudyInstanceUID")
    df["group"] = df.Report.map(
        lambda r: hashlib.md5(normalize(r).encode()).hexdigest()
    )
    df["lang"] = df.Report.map(detect_lang)

    groups = (
        df.groupby("group")
        .agg(n=("StudyInstanceUID", "size"), lang=("lang", "first"),
             **{lab: (lab, "mean") for lab in labels})
        .sample(frac=1, random_state=SEED)
    )

    assignment = assign_folds(groups, labels)
    df["fold"] = df.group.map(assignment)
    df[["StudyInstanceUID", "fold"]].to_csv(out_csv, index=False)

    print(df.fold.value_counts().sort_index().to_string())
    print("\nlang x fold:")
    print(pd.crosstab(df.lang, df.fold).to_string())
    print("\nper-fold mean weak-label prevalence (should be ~equal):")
    print(df.groupby("fold")[labels].mean().round(3).to_string())
    dup = df.groupby("group").fold.nunique()
    if not (dup == 1).all():
        raise ValueError(f"{int((dup != 1).sum())} report groups span more than one fold")
    print(f"\nOK: no group spans folds; wrote {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_CSV)
