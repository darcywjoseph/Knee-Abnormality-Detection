"""Build a per-(study, slot) slice-direction correction table for cache v3.

Some studies store a slot's stack back-to-front, so slot k's slice i means different
anatomy study to study. Each non-reference slot is compared, slice by slice, against its
same-plane reference slot (sagittal -> slot 0, coronal -> slot 1; axial has no partner)
and a flip flag is written when reversing the stack correlates better. Training applies
the flags, so the cache need not be rebuilt.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knee.common import CACHE_GLOB, SLOT_NAMES, slice_descriptors

REF = {3: 0, 5: 0, 4: 1}          # slot -> same-plane reference slot
MIN_MARGIN = 0.02                 # below this the forward/reversed call is noise; leave as-is
OUT_CSV = "data/slice_dir_fix.csv"


def study_flips(vol: np.ndarray, present: np.ndarray) -> dict[str, int]:
    """Decides, for one study, which slots are stored back-to-front.

    Args:
        vol: The cached volume, shape (slots, slices, H, W).
        present: Per-slot presence mask; a slot is only tested when it and its reference
            are both present.

    Returns:
        A `flip<slot>` flag for every slot in REF, 1 when reversing the stack correlates
        better with the same-plane reference by more than MIN_MARGIN.
    """
    desc: dict[int, np.ndarray] = {}
    flips = {}
    for slot, ref in REF.items():
        flips[f"flip{slot}"] = 0
        if not (present[slot] and present[ref]):
            continue
        for k in (slot, ref):
            if k not in desc:
                desc[k] = slice_descriptors(vol[k])
        fwd = float((desc[ref] * desc[slot]).sum(axis=1).mean())
        rev = float((desc[ref] * desc[slot][::-1]).sum(axis=1).mean())
        if rev - fwd > MIN_MARGIN:
            flips[f"flip{slot}"] = 1
    return flips


def main(cache_glob: str = CACHE_GLOB, out: str = OUT_CSV) -> None:
    """Scores every cached study and writes the flip table.

    Args:
        cache_glob: Glob over the cache .npz files. Nothing is written when it matches
            no file; the usage line is printed instead.
        out: Where to write the (StudyInstanceUID, flip3, flip4, flip5) table.
    """
    files = sorted(glob.glob(cache_glob))
    if not files:
        print(f"no cache files match {cache_glob!r}")
        print("usage: python -m knee.data.slice_directions [cache_glob] [out_csv]")
        return
    print(f"{len(files)} cache files")
    rows = []
    for i, f in enumerate(files):
        with np.load(f) as d:
            vol, present = d["vol"], d["present"].astype(bool)
        rows.append({"StudyInstanceUID": Path(f).stem, **study_flips(vol, present)})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    for slot in REF:
        print(f"  {SLOT_NAMES[slot]:18s} flipped {int(df[f'flip{slot}'].sum()):4d} of {len(df)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CACHE_GLOB,
         sys.argv[2] if len(sys.argv) > 2 else OUT_CSV)
