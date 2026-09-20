"""Shared scaffolding for the diagnostic figures: a headless pyplot and cache helpers.

Import `plt` from here rather than from matplotlib, so the Agg backend is always selected
before pyplot loads.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  - the backend must be set before pyplot loads

__all__ = ["plt", "cache_paths", "show_slice"]


def cache_paths(source: str) -> list[Path]:
    """Lists the cache .npz files under a directory or matching a glob.

    Args:
        source: A cache directory, searched recursively, or a glob over .npz files.

    Returns:
        The paths, sorted.
    """
    root = Path(source)
    if root.is_dir():
        return sorted(root.rglob("*.npz"))
    return sorted(Path(p) for p in glob.glob(source))


def show_slice(ax, image: np.ndarray) -> None:
    """Draws one greyscale slice on `ax`, tickless and on the stored 0-255 scale.

    Args:
        ax: The axes to draw into.
        image: One slice, shape (H, W), dtype uint8.
    """
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
