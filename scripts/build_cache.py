"""Builds one shard of the DICOM cache."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np

from knee.cache import (
    CacheConfig,
    build_cache,
    coverage_report,
    find_root,
    load_study_rows,
    pending_studies,
    shard_studies,
    write_manifest,
)


def parse_args():
    """Reads the command line.

    Returns:
        The parsed arguments; everything not listed comes from CacheConfig's defaults.
    """
    d = CacheConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comp-dir", default="data", help="competition mount, tried first")
    p.add_argument("--out-dir", default="data/cache", help="where the .npz files go")
    p.add_argument("--split", default="train", choices=("train", "test"))
    p.add_argument("--shard-index", type=int, default=d.shard_index)
    p.add_argument("--n-shards", type=int, default=d.n_shards)
    p.add_argument("--workers", type=int, default=d.n_workers, help="0 -> os.cpu_count()")
    p.add_argument("--budget-h", type=float, default=d.time_budget_h,
                   help="stop after this many hours; rerun to resume")
    p.add_argument("--limit", type=int, default=0,
                   help="build only the first N studies of the shard (0 -> all)")
    return p.parse_args()


def main():
    """Builds the shard and prints the coverage report."""
    args = parse_args()
    cfg = CacheConfig(n_shards=args.n_shards, shard_index=args.shard_index,
                      n_workers=args.workers, time_budget_h=args.budget_h)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    root = find_root(args.comp_dir, args.split)
    series_dir = root / f"{args.split}_series"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study_rows = load_study_rows(root, args.split)
    print(f"root: {root} | studies: {len(study_rows)}")

    studies = shard_studies(sorted(study_rows), cfg.n_shards, cfg.shard_index)
    if cfg.n_shards > 1:
        print(f"shard {cfg.shard_index}/{cfg.n_shards}: {len(studies)} studies")
    if args.limit:
        studies = studies[: args.limit]
        print(f"limit: {len(studies)} studies only")
    todo = pending_studies(studies, out_dir)
    print(f"{len(studies)} in scope | {len(studies) - len(todo)} cached | {len(todo)} to build")

    records, meta_rows, counts = build_cache(todo, study_rows, series_dir, out_dir, cfg)
    print(f"done | {counts}")

    manifest_df, meta_df, (man_path, meta_path) = write_manifest(
        records, meta_rows, out_dir, args.split, cfg.shard_index)
    print(f"manifest: {man_path}\nseries metadata: {meta_path}\n")
    print(coverage_report(manifest_df, meta_df, out_dir, len(studies), cfg))


if __name__ == "__main__":
    main()
