"""Figure: what "stored back-to-front" means.

Picks a study whose sagittal T1 slot the direction test flagged, and shows three rows of the
same eight slice positions: the reference sagittal FS stack, the sagittal T1 stack as the cache
stores it, and the same stack reversed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knee.common import SLOT_NAMES
from knee.eval.figure_base import cache_paths, plt, show_slice

SHOW = np.linspace(2, 21, 8).astype(int)
CACHE_DIR = "data/cache_v3"
FIX_CSV = "data/slice_dir_fix.csv"
DEFAULT_OUT = "docs/figures/slice_direction.png"
# One layout, applied before the block titles are positioned: a second subplots_adjust would
# move the axes out from under the text placed against their old positions.
LAYOUT = dict(left=0.13, right=0.99, top=0.93, bottom=0.07, hspace=0.12, wspace=0.03)


def pick(pool: set[str], files: dict[str, Path], slot: int, ref: int) -> tuple[str, np.ndarray]:
    """Finds the first cached study of `pool` that has both the slot and its reference.

    Args:
        pool: Candidate study UIDs.
        files: Study UID to cache .npz path.
        slot: The slot under test.
        ref: The same-plane reference slot.

    Returns:
        The chosen study UID and its cached volume.

    Raises:
        ValueError: No study in `pool` is cached with both slots present.
    """
    for uid in sorted(pool):
        path = files.get(uid)
        if path is None:
            continue
        with np.load(path) as d:
            if d["present"][slot] and d["present"][ref]:
                return uid, d["vol"]
    raise ValueError(f"no cached study with slots {slot} and {ref} present")


def main(cache_dir: str = CACHE_DIR, table: str = FIX_CSV, out: str = DEFAULT_OUT,
         slot: int = 5, ref: int = 0) -> None:
    """Draws the flagged / not-flagged comparison and writes it to `out`.

    Args:
        cache_dir: Cache directory or glob holding the .npz files.
        table: The slice-direction table.
        out: Output PNG path.
        slot: The slot whose direction is in question.
        ref: The same-plane reference slot it is compared against.
    """
    fix = pd.read_csv(table).set_index("StudyInstanceUID")
    files = {p.stem: p for p in cache_paths(cache_dir)}
    picks = [("FLAGGED — cache stores this one reversed",
              *pick(set(fix.index[fix[f"flip{slot}"] == 1]), files, slot, ref)),
             ("NOT FLAGGED — stored the right way round",
              *pick(set(fix.index[fix[f"flip{slot}"] == 0]), files, slot, ref))]

    fig, axes = plt.subplots(6, len(SHOW), figsize=(len(SHOW) * 1.5, 10.4))
    fig.subplots_adjust(**LAYOUT)
    for b, (_, _, vol) in enumerate(picks):
        rows = [(f"{SLOT_NAMES[ref]}  (reference)", vol[ref][SHOW]),
                (f"{SLOT_NAMES[slot]}  as stored", vol[slot][SHOW]),
                (f"{SLOT_NAMES[slot]}  reversed", vol[slot][::-1][SHOW])]
        for r, (label, images) in enumerate(rows):
            for c in range(len(SHOW)):
                ax = axes[b * 3 + r, c]
                show_slice(ax, images[c])
                if c == 0:
                    ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
                if b * 3 + r == 0:
                    ax.set_title(f"slice {SHOW[c]}", fontsize=8)
            for spine in axes[b * 3 + r, 0].spines.values():
                spine.set_visible(False)
    # Drawn as figure text, after the final layout, so a block title cannot land on an axes.
    for b, (title, _, _) in enumerate(picks):
        top = axes[b * 3, 0].get_position().y1
        fig.text(0.13, top + 0.012, title, fontsize=11, fontweight="bold", ha="left",
                 va="bottom", color="#b00020" if b == 0 else "#0a6b2e")
    fig.suptitle('What "stored back-to-front" means: the same slice positions, two studies',
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005,
             "Top study: the middle row walks through the knee the OPPOSITE way from the "
             "reference, so slot 5's slice 10 is not\nthe same anatomy as slot 0's slice 10. "
             "The bottom row puts it back in step. Lower study: already in step, left alone.",
             ha="center", fontsize=9, color="0.3")
    fig.savefig(out, dpi=115)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
