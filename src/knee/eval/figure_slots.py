"""Figure: a cached study as six slots by many slices, and what depth sampling drops."""
from __future__ import annotations

import sys

import numpy as np
from matplotlib.patches import Rectangle

from knee.common import CACHE_GLOB, SLOT_NAMES
from knee.eval.figure_base import cache_paths, plt, show_slice

# Slice stride, its description, and the colour panel C draws it in.
STRIDES = [(3, "every 3rd slice", "#c0392b"),
           (2, "every 2nd slice", "#1e8449"),
           (1, "every slice", "#2471c9")]
N_COL = 8
SCAN_LIMIT = 400
DEFAULT_OUT = "docs/figures/cache_slots_and_depth.png"


def pick_study(files: list) -> tuple[str, np.ndarray, np.ndarray, str]:
    """Picks a typical cached study: most slots present, ideally one missing.

    The corpus averages 4.6 of 6 slots, so a study with exactly one empty slot shows both what
    a slot looks like and what an absent one looks like. The thresholds are relative to the
    cache's slot count, so a cache with fewer slots still yields a study.

    Args:
        files: Cache .npz paths, searched in order up to SCAN_LIMIT entries.

    Returns:
        The chosen file, its volume, its presence mask and its laterality ('unknown' when the
        cache file does not record one).

    Raises:
        ValueError: No file was given, or none has enough slots present.
    """
    if not files:
        raise ValueError("no cache files to choose from")
    best = None
    for f in files[:SCAN_LIMIT]:
        with np.load(f) as d:
            n_slot = int(d["present"].shape[0])
            n_present = int(d["present"].sum())
            if n_present < n_slot - 2:
                continue
            side = str(d["side"]) if "side" in d.files else "unknown"
            study = (str(f), d["vol"], d["present"], side)
        if n_present == n_slot - 1:
            return study
        if best is None:
            best = study
    if best is None:
        raise ValueError(f"no study among the first {SCAN_LIMIT} files has enough slots")
    return best


def draw_slots(fig, gs, vol: np.ndarray, present: np.ndarray, side: str) -> None:
    """Draws panel A: every slot as a row, N_COL depths as columns.

    Args:
        fig: The figure to draw into.
        gs: The gridspec slot panel A occupies.
        vol: The study volume, shape (slots, slices, H, W).
        present: Per-slot presence mask.
        side: The study's laterality, for the caption.
    """
    n_slot, n_slice = vol.shape[0], vol.shape[1]
    show = np.linspace(0, n_slice - 1, N_COL).astype(int)
    gsa = gs.subgridspec(n_slot, N_COL, wspace=0.03, hspace=0.06)
    row_axes = []
    for r in range(n_slot):
        row_axes.append([])
        for c, s in enumerate(show):
            ax = fig.add_subplot(gsa[r, c])
            row_axes[r].append(ax)
            show_slice(ax, vol[r, s])
            for spine in ax.spines.values():
                spine.set_color("#bbb")
                spine.set_linewidth(0.5)
            if r == 0:
                ax.set_title(f"slice {s}", fontsize=9.5, color="#555", pad=4)
            if c == 0:
                ax.set_ylabel(SLOT_NAMES[r], fontsize=11, rotation=0, ha="right", va="center",
                              labelpad=14, color="#111" if present[r] else "#b00")
            if not present[r]:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor="white", alpha=0.75, zorder=3))
    for r in range(n_slot):
        if not present[r]:
            first = row_axes[r][0].get_position()
            last = row_axes[r][-1].get_position()
            fig.text((first.x0 + last.x1) / 2, (first.y0 + first.y1) / 2,
                     "this study has no such scan  —  slot zero-filled and flagged absent",
                     ha="center", va="center", fontsize=12.5, color="#b00", weight="bold",
                     zorder=20)
    fig.text(0.5, 0.975,
             f"A.   One cached study  =  {n_slot} slots  x  {n_slice} slices  "
             f"x  {vol.shape[-1]} px",
             ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.953,
             "Each ROW is a separate scan of the same knee — a slot: one plane through the "
             "joint, one tissue contrast.   Each COLUMN steps deeper through that scan.\n"
             f"Left and right knees are mirrored to a common side (this study: {side}).   "
             "Scanners and protocols differ, so slots are assigned by reading DICOM headers.",
             ha="center", fontsize=11, color="#444")


def draw_strip(fig, gs, vol: np.ndarray) -> None:
    """Draws panel B: every stored slice of slot 0 as one strip.

    Args:
        fig: The figure to draw into.
        gs: The gridspec slot panel B occupies.
        vol: The study volume, shape (slots, slices, H, W).
    """
    n_slice = vol.shape[1]
    gsb = gs.subgridspec(1, n_slice, wspace=0.025)
    strip_ax = None
    for s in range(n_slice):
        ax = fig.add_subplot(gsb[0, s])
        strip_ax = strip_ax or ax
        show_slice(ax, vol[0, s])
        for spine in ax.spines.values():
            spine.set_color("#bbb")
            spine.set_linewidth(0.4)
        ax.set_title(str(s), fontsize=7.5, color="#666", pad=2)
    fig.text(0.088, strip_ax.get_position().y1 + 0.028,
             f"B.   Depth sampling — all {n_slice} stored slices of the {SLOT_NAMES[0]} slot, "
             "and which ones each run feeds the model",
             ha="left", fontsize=14.5, weight="bold")


def draw_strides(fig, gs, n_slice: int) -> None:
    """Draws panel C: which slices each stride keeps.

    Args:
        fig: The figure to draw into.
        gs: The gridspec slot panel C occupies.
        n_slice: How many slices the cache stores per slot.
    """
    axc = fig.add_subplot(gs)
    for k, (stride, label, colour) in enumerate(STRIDES):
        used = np.arange(0, n_slice, stride)
        axc.scatter(used, [k] * len(used), s=105, marker="s", color=colour)
        missed = np.setdiff1d(np.arange(n_slice), used)
        axc.scatter(missed, [k] * len(missed), s=105, marker="s", facecolor="none",
                    edgecolor="#d5d5d5", linewidth=0.9)
        axc.text(n_slice + 0.4, k, f"   {label}", va="center", fontsize=10.5, color=colour,
                 weight="bold")
    axc.set_yticks(range(len(STRIDES)))
    axc.set_yticklabels(
        [f"{len(range(0, n_slice, stride))} slices/slot\n"
         f"{len(range(0, n_slice, stride)) * len(SLOT_NAMES)} images/study"
         for stride, _, _ in STRIDES], fontsize=10)
    axc.set_xticks(range(n_slice))
    axc.set_xticklabels(range(n_slice), fontsize=7.5, color="#666")
    axc.set_xlim(-0.6, n_slice + 7.0)
    axc.set_ylim(-0.6, len(STRIDES) - 0.4)
    axc.invert_yaxis()
    for spine in axc.spines.values():
        spine.set_visible(False)
    axc.tick_params(length=0)


def main(out: str = DEFAULT_OUT, cache_glob: str = CACHE_GLOB) -> None:
    """Builds the three-panel figure and writes it to `out`.

    Args:
        out: Output PNG path.
        cache_glob: Cache directory or glob over the .npz files.
    """
    f, vol, present, side = pick_study(cache_paths(cache_glob))

    fig = plt.figure(figsize=(16, 14.4), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[6.0, 0.80, 0.70], hspace=0.20)
    draw_slots(fig, gs[0], vol, present, side)
    draw_strip(fig, gs[1], vol)
    draw_strides(fig, gs[2], vol.shape[1])

    fig.text(0.088, 0.038,
             "A meniscal tear or a fracture line can be visible on only two or three adjacent "
             "slices, so skipping slices can step straight past the finding: the denser the "
             "sampling, the less the model can miss.",
             fontsize=11, color="#444")
    fig.subplots_adjust(left=0.115, right=0.985, top=0.935, bottom=0.105)
    fig.savefig(out, dpi=112, facecolor="white")
    print("wrote", out, "| study", f, "| present", present.tolist(), "| side", side)


if __name__ == "__main__":
    main(*sys.argv[1:3])
