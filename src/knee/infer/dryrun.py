"""Dry run of the submission kernel's inference half, on a laptop.

The inference cells are pulled out of the submission notebook by marker string and executed
verbatim, so this script cannot drift from the kernel. It exercises what would otherwise only
fail on the hidden test: the strict checkpoint load, slice_flips, load_study, the TTA loop, the
rank-mean and the submission writer with its sanity checks. With --notebook it instead runs
every code cell of a given notebook, which is how a bundled standalone kernel is checked.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from knee.common import TARGETS, per_label_auc

REPO = Path(__file__).resolve().parents[3]
NOTEBOOK = REPO / "notebooks" / "submissions" / "submition-01.ipynb"
WEAK_CSV = REPO / "data" / "weak_labels" / "train_weak_labels.csv"

# One marker per inference cell, in execution order. Each must match exactly one code cell.
MARKERS = (
    "class AttnPool",
    "def find_checkpoints",
    "def slice_flips",
    "def load_study",
    "studies to predict",
    "final[ok] = rank_mean",
    "distinct values per label",
)

SYNTHETIC_STUDIES = 8
SYNTHETIC_SLOTS = 6
SYNTHETIC_SLICES = 24
SYNTHETIC_IMG = 320


def build_synthetic_cache(out_dir: Path, n_studies: int, seed: int = 0) -> list[str]:
    """Writes cache-v3-shaped .npz files with smoothly varying random volumes.

    The content is meaningless, but the shapes, dtypes and the presence mask are exactly what
    the builder emits, which is all the inference path reads.

    Args:
        out_dir: Directory to write <uid>.npz into; created if missing.
        n_studies: How many studies to write.
        seed: Seed for the volume generator.

    Returns:
        The sorted StudyInstanceUIDs written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    small_shape = (SYNTHETIC_SLOTS, SYNTHETIC_SLICES, SYNTHETIC_IMG // 8, SYNTHETIC_IMG // 8)
    uids = []
    for i in range(n_studies):
        uid = f"1.2.826.0.1.TEST.{i:04d}"
        base = rng.random((SYNTHETIC_SLOTS, 1, SYNTHETIC_IMG // 8, SYNTHETIC_IMG // 8))
        drift = np.linspace(0.0, 1.0, SYNTHETIC_SLICES)[None, :, None, None]
        small = base * (1.0 - drift) + rng.random(small_shape) * drift
        vol = np.repeat(np.repeat(small, 8, axis=2), 8, axis=3)
        vol = (vol * 255).astype(np.uint8)
        present = np.ones(SYNTHETIC_SLOTS, np.uint8)
        present[-1] = i % 2            # one sometimes-missing slot, as in the real cache
        vol[present == 0] = 0
        np.savez_compressed(out_dir / f"{uid}.npz", vol=vol, present=present)
        uids.append(uid)
    return sorted(uids)


def notebook_cells(path: Path) -> list[tuple[str, str]]:
    """Reads every code cell of a notebook, in order.

    This is how a bundled kernel is checked: its cells are self-contained, so they run in one
    namespace with no patching beyond the CFG override the bundler leaves in place.

    Args:
        path: The notebook to read.

    Returns:
        (label, source) pairs, in execution order.
    """
    with open(path, encoding="utf-8") as f:
        notebook = json.load(f)
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    return [(f"cell {i}", s) for i, s in enumerate(sources)]


def inference_cells(work_dir: Path, tta_passes: int) -> list[tuple[str, str]]:
    """Extracts the kernel's inference cells in order, with its Kaggle paths rewritten.

    Args:
        work_dir: Directory standing in for /kaggle/working.
        tta_passes: Number of TTA offsets to keep; 1 halves the dry run's wall clock.

    Returns:
        (marker, patched source) pairs, in execution order.

    Raises:
        ValueError: A marker matches no code cell, or more than one.
    """
    with open(NOTEBOOK, encoding="utf-8") as f:
        notebook = json.load(f)
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]

    out = []
    for marker in MARKERS:
        hits = [c for c in cells if marker in c]
        if len(hits) != 1:
            raise ValueError(f"marker {marker!r} matched {len(hits)} code cells in "
                             f"{NOTEBOOK.name}; the kernel has changed")
        source = hits[0].replace("/kaggle/working", str(work_dir))
        if marker == "def load_study" and tta_passes == 1:
            source = source.replace("TTA_OFFSETS = [0, 1]",
                                    "TTA_OFFSETS = [0]  # dry run: one pass")
        out.append((marker, source))
    return out


def score_against_weak(submission_path: Path) -> float | None:
    """Scores a dry-run submission against the weak labels, when they cover its studies.

    Being in-fold for most checkpoints, this is a smoke test for a broken prediction path, not
    a score.

    Args:
        submission_path: The submission.csv written by the kernel cells.

    Returns:
        The mean AUC over the scorable labels, or None when nothing could be scored.
    """
    submission = pd.read_csv(submission_path)
    weak = pd.read_csv(WEAK_CSV).set_index("StudyInstanceUID")
    known = submission[submission.StudyInstanceUID.isin(weak.index)]
    if len(known) < 10:
        print(f"only {len(known)} studies carry weak labels; skipping the AUC check")
        return None
    y = weak.loc[known.StudyInstanceUID, TARGETS].ge(0.5).astype(int)
    aucs = per_label_auc(y, known)
    mean_auc = float(np.mean(list(aucs.values())))
    print(f"\nweak AUC over {len(known)} studies, {len(aucs)}/12 labels = {mean_auc:.4f}")
    print("NOTE: in-fold for most checkpoints -- a smoke test for a broken path, not a score.")
    return mean_auc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Argument list, or None to read sys.argv.

    Returns:
        The parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", type=Path, required=True, help="directory of best_fold*.pt")
    p.add_argument("--cache", type=Path, help="cache-v3 directory of <study>.npz files")
    p.add_argument("--synthetic", action="store_true",
                   help="build a synthetic cache instead of reading --cache")
    p.add_argument("--studies", type=int, default=0, help="cap on studies to predict; 0 = all")
    p.add_argument("--tta", type=int, default=1, choices=(1, 2),
                   help="TTA passes; 1 (default) halves the wall clock")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/infer_dryrun"),
                   help="stands in for /kaggle/working")
    p.add_argument("--notebook", type=Path,
                   help="run every code cell of this notebook instead of the kernel's "
                        "inference cells; for checking a bundled kernel")
    p.add_argument("--score", action="store_true",
                   help="also score the result against the weak labels")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs the kernel's inference cells against a local cache and weights directory.

    Args:
        argv: Argument list, or None to read sys.argv.

    Returns:
        0 on success.

    Raises:
        FileNotFoundError: The cache or weights directory does not exist.
        ValueError: Neither --cache nor --synthetic was given, or a marker no longer matches
            exactly one cell.
    """
    args = parse_args(argv)
    started = time.time()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.weights.resolve()
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"no weights directory at {weights_dir}")

    if args.synthetic:
        cache_dir = (args.cache or work_dir / "cache").resolve()
        uids = build_synthetic_cache(cache_dir, SYNTHETIC_STUDIES)
        print(f"synthetic cache: {len(uids)} studies in {cache_dir}")
    elif args.cache is None:
        raise ValueError("give --cache DIR, or --synthetic to build one")
    else:
        cache_dir = args.cache.resolve()
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"no cache directory at {cache_dir}")
        uids = sorted(p.stem for p in cache_dir.rglob("*.npz"))
        print(f"cache: {len(uids)} studies in {cache_dir}")
    if args.studies:
        uids = uids[:args.studies]

    if args.notebook:
        cells = notebook_cells(args.notebook)
        namespace = {
            "__name__": "__main__",
            "CFG_OVERRIDE": {"comp_dir": str(work_dir), "weights": str(weights_dir),
                             "work_dir": str(work_dir), "tta": args.tta},
        }
        for label, source in cells:
            print(f"\n----- {label} -----", flush=True)
            exec(compile(source, f"<{label}>", "exec"), namespace)
        submission = work_dir / "submission.csv"
        print(f"\nwrote {submission}")
        if args.score:
            score_against_weak(submission)
        print(f"DRY RUN PASSED in {(time.time() - started) / 60:.1f} min")
        return 0

    cells = inference_cells(work_dir, args.tta)

    # find_checkpoints() searches "/kaggle/input", "weights", "runs" and "." relative to the
    # working directory, and insists on exactly one directory holding a full fold set. Running
    # from inside the chosen weights directory is what pins it to those checkpoints.
    namespace = {
        "__name__": "__main__",
        "Path": Path, "np": np, "pd": pd, "time": time,
        "NB_T0": started,
        "CFG": {"OUT_DIR": str(cache_dir)},
        # ROOT is only read for sample_submission.csv; without one the kernel falls back to
        # STUDY_ROWS, which is the list of studies we want predicted.
        "ROOT": work_dir,
        "STUDY_ROWS": {u: [] for u in uids},
    }
    cwd = Path.cwd()
    os.chdir(weights_dir)
    try:
        for marker, source in cells:
            print(f"\n----- cell: {marker} -----", flush=True)
            exec(compile(source, f"<{marker}>", "exec"), namespace)
    finally:
        os.chdir(cwd)

    submission = work_dir / "submission.csv"
    print(f"\nwrote {submission}")
    if args.score:
        score_against_weak(submission)
    print(f"DRY RUN PASSED in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
