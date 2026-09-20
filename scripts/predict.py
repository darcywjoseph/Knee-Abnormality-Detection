"""Builds the test cache if needed, then writes submission.csv.

An existing cache under <work-dir>/cache is reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from knee.cache import CacheConfig, build_cache, find_root, load_study_rows, pending_studies
from knee.predict import InferConfig, find_checkpoints, load_models, write_submission


def parse_args():
    """Reads the command line.

    Returns:
        The parsed arguments; everything not listed comes from InferConfig's defaults.
    """
    d = InferConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comp-dir", default="data", help="competition mount, tried first")
    p.add_argument("--weights", required=True, help="directory holding best_fold*.pt")
    p.add_argument("--work-dir", default="/tmp/predict",
                   help="holds cache/ and the submission.csv that is written")
    p.add_argument("--tta", type=int, default=len(d.tta_offsets), help="TTA passes")
    p.add_argument("--budget-h", type=float, default=d.budget_h,
                   help="wall clock the whole run gets")
    return p.parse_args()


def main():
    """Builds any missing cache, loads the fold models and writes the submission."""
    args = parse_args()
    work_dir = Path(args.work_dir)
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = InferConfig(comp_dir=args.comp_dir, out_dir=str(cache_dir),
                      tta_offsets=tuple(range(args.tta)), budget_h=args.budget_h)

    try:
        root = find_root(cfg.comp_dir, cfg.split)
    except FileNotFoundError:
        root = None
    if root is None:
        studies = sorted(p.stem for p in cache_dir.glob("*.npz"))
        print(f"no competition mount; predicting the {len(studies)} cached studies")
    else:
        study_rows = load_study_rows(root, cfg.split)
        studies = sorted(study_rows)
        todo = pending_studies(studies, cache_dir)
        print(f"root: {root} | {len(studies)} studies | {len(todo)} to cache")
        if todo:
            _, _, counts = build_cache(todo, study_rows, root / f"{cfg.split}_series",
                                       cache_dir, CacheConfig(n_shards=1))
            print(f"cache done | {counts}")

    ckpts = find_checkpoints(cfg.n_folds, bases=[Path(args.weights)])
    models = load_models(ckpts, cfg)
    print(f"{len(models)} models loaded from {ckpts[0].parent} | device {cfg.device}")
    write_submission(models, studies, work_dir / "submission.csv", cfg)
    print(f"wrote {work_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
