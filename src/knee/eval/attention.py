"""Diagnostic: what does the slice-attention actually attend to?

AttnPool returns a softmax over the slices of each slot. If it is near-uniform, pooling is not
selecting and any gain from storing more slices is about coverage; if it concentrates, the plot
shows where in the joint the model looks, and whether the band crop is cutting anything useful.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

from knee.common import CACHE_GLOB, SLOT_NAMES
from knee.eval.figure_base import plt
from knee.model import Model2p5D

STRIDE = 2
ATTN_DIM = 128
MAX_STUDIES = 240
FOLDS_CSV = "data/folds.csv"
DEFAULT_OUT = "docs/figures/attention_by_slice.png"

# The diagnostic only makes sense for the shared-gate AttnPool head on a CNN backbone; the
# label-attention head has no single per-slice softmax to plot.
DIAG_CFG = {
    "BACKBONE": "convnext_tiny", "PRETRAINED": False, "VIT_IMG": 0, "VIT_UNFREEZE": 0,
    "GRAD_CHECKPOINT": False, "N_SLOTS": 6, "HEAD": "attn", "HEAD_HIDDEN": 0,
    "SLOT_EMB": False, "ATTN_DIM": ATTN_DIM, "DROPOUT": 0.0,
}


def load_study(path: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Loads one cached study as the model's input tensor.

    Args:
        path: The study's .npz cache file.
        idx: Slice indices to take from every slot.

    Returns:
        Images of shape (slots, len(idx), 3, H, W), normalised the way training does, and the
        per-slot presence mask as float32.
    """
    with np.load(path) as d:
        vol, present = d["vol"], d["present"].astype(np.float32)
    n = vol.shape[1]
    x = np.stack([vol[:, np.clip(idx - 1, 0, n - 1)],
                  vol[:, idx],
                  vol[:, np.clip(idx + 1, 0, n - 1)]], axis=2).astype(np.float32) / 255.0
    return (x - 0.449) / 0.226, present


def attention_of(model: Model2p5D, x: torch.Tensor) -> torch.Tensor:
    """Runs the backbone and the pooling gate, returning the per-slice attention.

    Model2p5D.forward drops the attention weights, so this repeats its non-ViT path and keeps
    them.

    Args:
        model: A Model2p5D built with HEAD == 'attn' and a CNN backbone.
        x: Images of shape (B, slots, S, 3, H, W).

    Returns:
        Attention weights of shape (B, slots, S), each slot's slices summing to 1.
    """
    b, k, s = x.shape[:3]
    f = model.backbone(x.reshape(b * k * s, *x.shape[3:])).reshape(b * k, s, -1)
    _, a = model.pool(f)
    return a.reshape(b, k, s)


def collect(ckpt: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads a checkpoint and records its attention over one fold's validation studies.

    Args:
        ckpt: Path to a training checkpoint (raw state dict, or one under 'model' /
            'state_dict').
        fold: Which fold's validation studies to run.

    Returns:
        The attention array of shape (N, slots, len(idx)), the (N, slots) presence array and
        the slice indices that were fed to the model.

    Raises:
        ValueError: No cached study of that fold was found.
    """
    files = {os.path.basename(f)[:-4]: f for f in glob.glob(CACHE_GLOB)}
    folds = pd.read_csv(FOLDS_CSV)
    val = [u for u in folds.loc[folds.fold == fold, "StudyInstanceUID"] if u in files]
    val = val[:MAX_STUDIES]
    if not val:
        raise ValueError(f"no cached study of fold {fold} under {CACHE_GLOB}")

    with np.load(files[val[0]]) as d:
        n_slot, n_slice = d["vol"].shape[0], d["vol"].shape[1]
    idx = np.arange(0, n_slice, STRIDE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model2p5D({**DIAG_CFG, "N_SLOTS": n_slot}).to(device).eval()
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt} | missing {len(missing)} unexpected {len(unexpected)}")

    attn, pres = [], []
    with torch.no_grad():
        for uid in val:
            x, p = load_study(files[uid], idx)
            batch = torch.from_numpy(x).unsqueeze(0).to(device)
            attn.append(attention_of(model, batch)[0].cpu().numpy())
            pres.append(p)
    return np.array(attn), np.array(pres), idx


def plot(attn: np.ndarray, pres: np.ndarray, idx: np.ndarray, fold: int, out: str) -> None:
    """Draws one panel per slot: mean attention against slice index.

    Args:
        attn: Attention of shape (N, slots, S).
        pres: Presence mask of shape (N, slots).
        idx: The slice indices the attention is over.
        fold: Fold number, for the title.
        out: Output PNG path.
    """
    uniform = 1.0 / len(idx)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), facecolor="white")
    for k, ax in enumerate(axes.ravel()):
        m = pres[:, k] > 0
        if not m.any():
            ax.set_title(f"{SLOT_NAMES[k]} — never present")
            ax.axis("off")
            continue
        a = attn[m, k]
        ax.plot(idx, a.mean(0), color="#2471c9", lw=2.2, label="mean attention")
        ax.fill_between(idx, np.percentile(a, 10, axis=0), np.percentile(a, 90, axis=0),
                        color="#2471c9", alpha=0.16, label="10-90th pct")
        ax.axhline(uniform, color="#b00", ls="--", lw=1.3, label=f"uniform ({uniform:.3f})")
        ax.set_title(f"{SLOT_NAMES[k]}   n={int(m.sum())}   "
                     f"peak/uniform = {a.max(1).mean() / uniform:.2f}x", fontsize=11.5)
        ax.set_xlabel("slice index (0 = one edge of the scan, 23 = the other)", fontsize=9)
        ax.set_ylabel("attention weight", fontsize=9)
        ax.set_ylim(bottom=0)
        if k == 0:
            ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Where the slice-attention looks, per slot "
                 f"(fold {fold} validation, n={len(attn)} studies)",
                 fontsize=14.5, weight="bold")
    fig.text(0.5, 0.005,
             "A flat line at the red dashed level means attention is uniform — the pooling "
             "layer is not selecting, and denser sampling helps through coverage alone.\n"
             "A peak means it has learned where in the joint to look; a peak hard against the "
             "first or last slice means the band crop is probably cutting useful anatomy.",
             ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])
    fig.savefig(out, dpi=115, facecolor="white")
    print("wrote", out)


def summarise(attn: np.ndarray, pres: np.ndarray, idx: np.ndarray) -> None:
    """Prints peak/uniform, argmax slice and entropy per slot.

    Args:
        attn: Attention of shape (N, slots, S).
        pres: Presence mask of shape (N, slots).
        idx: The slice indices the attention is over.
    """
    uniform = 1.0 / len(idx)
    for k in range(attn.shape[1]):
        m = pres[:, k] > 0
        if not m.any():
            continue
        a = attn[m, k]
        entropy = float(-(a * np.log(a + 1e-9)).sum(1).mean())
        print(f"  {SLOT_NAMES[k]:16s} peak/uniform {a.max(1).mean() / uniform:5.2f}x  "
              f"argmax slice {idx[a.mean(0).argmax()]:2d}  entropy {entropy:.3f} "
              f"(uniform {np.log(len(idx)):.3f})")


def main(argv: list[str]) -> int:
    """Runs the diagnostic end to end.

    Args:
        argv: Command-line arguments without the program name: a checkpoint path, then
            optionally a fold number and an output PNG path.

    Returns:
        A process exit status: 0 on success, 2 on a usage or data error.
    """
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python -m knee.eval.attention CHECKPOINT.pt [fold] [out.png]\n"
              f"       fold defaults to 0, out.png to {DEFAULT_OUT}", file=sys.stderr)
        return 2
    ckpt, fold = argv[0], int(argv[1]) if len(argv) > 1 else 0
    out = argv[2] if len(argv) > 2 else DEFAULT_OUT
    try:
        attn, pres, idx = collect(ckpt, fold)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("studies", len(attn), "| attention array", attn.shape)
    plot(attn, pres, idx, fold, out)
    summarise(attn, pres, idx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
