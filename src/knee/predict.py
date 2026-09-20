"""Inference: cached studies in, `submission.csv` out.

Loads one checkpoint per fold, scores every cached study under TTA, averages the folds by
per-label rank and writes the submission. The time budget is sized from a probe over the first
studies, so a slow machine drops TTA passes and then folds rather than running out of clock.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from knee.common import TARGETS, slice_descriptors
from knee.model import Model2p5D

__all__ = [
    "InferConfig", "REF_SLOT", "MIN_MARGIN", "find_checkpoints", "load_models", "slice_flips",
    "load_study", "predict", "rank_mean", "fit_budget", "write_submission",
]

# Slot -> same-plane reference slot. Some slots are stored back to front relative to their
# reference, which upstream preprocessing cannot always fix.
REF_SLOT = {3: 0, 5: 0, 4: 1}
MIN_MARGIN = 0.02

# Directories searched for checkpoints, in order; a Kaggle dataset mount wins.
CKPT_BASES = (Path("/kaggle/input"), Path("weights"), Path("runs"), Path("."))

MODEL_CFG = {
    "BACKBONE": "convnext_tiny",
    "N_SLOTS": 6,
    "ATTN_DIM": 128,
    "DROPOUT": 0.2,
    "SLICE_STRIDE": 2,
    "IMG": 320,
    # Inference-only settings, fixed: the architecture must match the checkpoints exactly.
    "PRETRAINED": False,
    "GRAD_CHECKPOINT": False,
    "HEAD": "attn_pool",
}


def _default_device() -> str:
    """Names the device to run on.

    Returns:
        'cuda' when a GPU is visible, else 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InferConfig:
    """Everything the inference stage needs beyond the checkpoints themselves.

    The cache-build settings are not repeated here: `knee.cache.CacheConfig` owns them, and
    only the slot count and image size, which this stage reads from `model_cfg`, are shared.

    Attributes:
        comp_dir: Competition mount, tried before the standard Kaggle locations.
        split: Which split to predict.
        out_dir: Cache directory holding one <study>.npz per study.
        model_cfg: Architecture of the checkpoints; must match how they were trained.
        n_folds: Fold checkpoints that make up one complete run.
        tta_offsets: Slice-sampling offsets, one forward pass each.
        reserve_s: Seconds held back from the budget for writing the submission.
        budget_h: Wall clock the whole kernel gets.
        device: Torch device string.
        seed: Seed for the random and numpy generators.
    """

    comp_dir: str = "/kaggle/input/rsna-knee-abnormality-detection"
    split: str = "test"
    out_dir: str = "/tmp/cache"
    model_cfg: dict = field(default_factory=lambda: dict(MODEL_CFG))
    n_folds: int = 5
    tta_offsets: tuple[int, ...] = (0, 1)
    reserve_s: int = 900
    budget_h: float = 9.0
    device: str = field(default_factory=_default_device)
    seed: int = 42


def find_checkpoints(n_folds: int = 5, bases: list[Path] | None = None) -> list[Path]:
    """Finds the fold checkpoints of one training run.

    Searches for `best_fold*.pt` and groups the hits by directory. Only a directory holding
    all `n_folds` folds is usable, so a half-uploaded weights dataset fails loudly here
    rather than quietly scoring with three folds.

    Args:
        n_folds: How many folds a complete run has.
        bases: Directories to search, or None for the standard ones.

    Returns:
        The fold checkpoint paths, ordered by fold number.

    Raises:
        SystemExit: No directory holds a complete set of folds, or more than one does.
    """
    by_dir = {}
    for base in list(CKPT_BASES if bases is None else bases):
        if Path(base).is_dir():
            for path in sorted(Path(base).rglob("best_fold*.pt")):
                by_dir.setdefault(path.parent, {})[path.stem.replace("best_fold", "")] = path
    complete = {d: v for d, v in by_dir.items() if len(v) == n_folds}
    if len(complete) != 1:
        raise SystemExit(
            f"need exactly one directory with all {n_folds} folds, found {len(complete)}: "
            f"{ {str(d): sorted(v) for d, v in by_dir.items()} }")
    folds = next(iter(complete.values()))
    return [folds[k] for k in sorted(folds)]


def load_models(paths, cfg: InferConfig) -> list[Model2p5D]:
    """Builds and loads one model per checkpoint, in eval mode.

    Args:
        paths: Fold checkpoint paths.
        cfg: The run's configuration; reads `model_cfg` and `device`.

    Returns:
        The loaded models.
    """
    models = []
    for path in paths:
        model = Model2p5D(cfg.model_cfg).to(cfg.device).eval()
        state = torch.load(path, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=True)
        models.append(model)
    return models


def slice_flips(vol, present, ref_slot=REF_SLOT, min_margin=MIN_MARGIN) -> list[int]:
    """Finds the slots stored back to front relative to their same-plane reference.

    Args:
        vol: The study's cache array, shape (slots, S, H, W).
        present: Per-slot presence flags.
        ref_slot: Slot index to reference slot index.
        min_margin: How much better the reversed correlation must be before flipping.

    Returns:
        Slot indices whose slice order should be reversed.
    """
    flips, desc = [], {}
    for slot, ref in ref_slot.items():
        if not (present[slot] and present[ref]):
            continue
        for k in (slot, ref):
            if k not in desc:
                desc[k] = slice_descriptors(vol[k])
        forward = float((desc[ref] * desc[slot]).sum(axis=1).mean())
        reverse = float((desc[ref] * desc[slot][::-1]).sum(axis=1).mean())
        if reverse - forward > min_margin:
            flips.append(slot)
    return flips


def load_study(study: str, offsets, cfg: InferConfig):
    """Loads one cached study as a batch of TTA views.

    Args:
        study: StudyInstanceUID.
        offsets: Slice-sampling offsets, one view each.
        cfg: The run's configuration; reads `out_dir` and the model's slice stride.

    Returns:
        A (views, present) pair, where views has shape (len(offsets), slots, S, 3, H, W), or
        None if the study has no cache file.
    """
    path = Path(cfg.out_dir) / f"{study}.npz"
    if not path.is_file():
        return None
    with np.load(path) as d:
        vol, present = d["vol"], d["present"].astype(np.float32)
    vol = vol.copy()
    for slot in slice_flips(vol, present):
        vol[slot] = vol[slot][::-1]

    n = vol.shape[1]
    base = np.arange(0, n, cfg.model_cfg["SLICE_STRIDE"])
    views = []
    for off in offsets:
        idx = np.clip(base + off, 0, n - 1)
        # Each sampled slice becomes a 3-channel image with its neighbours, hence 2.5D.
        x = np.stack([vol[:, np.clip(idx - 1, 0, n - 1)],
                      vol[:, idx],
                      vol[:, np.clip(idx + 1, 0, n - 1)]], axis=2).astype(np.float32) / 255.0
        views.append((x - 0.449) / 0.226)
    return np.stack(views), present


@torch.no_grad()
def predict(models, studies, offsets, cfg: InferConfig) -> np.ndarray:
    """Runs the ensemble over a list of studies, averaging each model over its TTA views.

    Args:
        models: The loaded fold models.
        studies: StudyInstanceUIDs, in submission order.
        offsets: Slice-sampling offsets to average over.
        cfg: The run's configuration; reads `device` and what `load_study` needs.

    Returns:
        Probabilities of shape (len(models), len(studies), 12). Rows for studies with no
        cache file are NaN.
    """
    out = np.full((len(models), len(studies), len(TARGETS)), np.nan, np.float32)
    t0 = time.time()
    for i, study in enumerate(studies):
        loaded = load_study(study, offsets, cfg)
        if loaded is None:
            continue
        x, present = loaded
        xb = torch.from_numpy(x).to(cfg.device)
        pb = torch.from_numpy(present).to(cfg.device).unsqueeze(0).expand(len(offsets), -1)
        for m, model in enumerate(models):
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.device == "cuda"):
                logits = model(xb, pb)
            out[m, i] = torch.sigmoid(logits.float()).mean(dim=0).cpu().numpy()
        if (i + 1) % 100 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"  {i+1}/{len(studies)} | {rate:.2f} s/study | "
                  f"eta {rate * (len(studies) - i - 1) / 60:.1f} min", flush=True)
    return out


def rank_mean(preds: np.ndarray) -> np.ndarray:
    """Averages several models' predictions by per-label rank.

    Separately trained folds share no output scale, so their probabilities are averaged as
    ranks; the metric only reads the ordering within a label anyway.

    Args:
        preds: Probabilities of shape (models, studies, 12), finite.

    Returns:
        Mean ranks in (0, 1), shape (studies, 12).
    """
    n_studies = preds.shape[1]
    return rankdata(preds, axis=1).mean(axis=0) / (n_studies + 1.0)


def fit_budget(unit_cost, n_studies, seconds_left, n_models, n_offsets) -> tuple[int, int]:
    """Sizes the ensemble to the wall clock left.

    TTA passes go first, then folds: a fold is worth more than a second view of the same
    study.

    Args:
        unit_cost: Seconds per study, per model, per TTA pass.
        n_studies: Studies still to predict.
        seconds_left: Wall clock remaining, minus the reserve.
        n_models: Models available.
        n_offsets: TTA passes wanted.

    Returns:
        The (models, offsets) counts to actually run, each at least 1.
    """
    def cost(models, offsets):
        """Projects the seconds a given ensemble size would take.

        Args:
            models: Models to run.
            offsets: TTA passes to run.

        Returns:
            Projected seconds.
        """
        return unit_cost * models * offsets * n_studies

    offsets = n_offsets
    while offsets > 1 and cost(n_models, offsets) > seconds_left:
        offsets -= 1
    models = n_models
    while models > 1 and cost(models, offsets) > seconds_left:
        models -= 1
    return models, offsets


def write_submission(models, studies, sub_path, cfg: InferConfig,
                     started: float | None = None) -> pd.DataFrame:
    """Probes the cost, predicts within budget, writes the submission and checks it.

    Args:
        models: The loaded fold models.
        studies: StudyInstanceUIDs, in submission order.
        sub_path: Where submission.csv is written.
        cfg: The run's configuration.
        started: Wall clock when the kernel started, or None to start it here.

    Returns:
        The submission, one row per study.

    Raises:
        ValueError: A label column came out constant, which scores 0.5 AUC.
    """
    started = time.time() if started is None else started
    offsets = list(cfg.tta_offsets)
    print(f"{len(studies)} studies to predict")

    probe = studies[:min(10, len(studies))]
    t0 = time.time()
    predict(models, probe, offsets, cfg)
    unit_cost = (time.time() - t0) / (len(probe) * len(models) * len(offsets))

    left = cfg.budget_h * 3600 - (time.time() - started) - cfg.reserve_s
    n_models, n_offsets = fit_budget(unit_cost, len(studies), left, len(models), len(offsets))
    print(f"probe: {unit_cost * len(models) * len(offsets):.2f} s/study | {left/3600:.2f} h left")
    if (n_models, n_offsets) != (len(models), len(offsets)):
        print(f"OVER BUDGET: trimmed to {n_models} models x {n_offsets} TTA")
    print(f"running {n_models} models x {n_offsets} TTA | "
          f"projected {unit_cost * n_models * n_offsets * len(studies) / 3600:.2f} h")

    preds = predict(models[:n_models], studies, offsets[:n_offsets], cfg)
    ok = np.isfinite(preds).all(axis=(0, 2))
    print(f"{ok.sum()}/{len(studies)} studies predicted; {(~ok).sum()} fall back to 0.5")

    final = np.full((len(studies), len(TARGETS)), 0.5, np.float32)
    if ok.any():
        final[ok] = rank_mean(preds[:, ok])
    out = pd.DataFrame(final, columns=TARGETS)
    out.insert(0, "StudyInstanceUID", studies)
    out.to_csv(sub_path, index=False)
    print(out.shape)

    # A constant column scores 0.5 AUC, and a code competition gives no second chance.
    chk = pd.read_csv(sub_path)
    nunique = chk[TARGETS].nunique()
    print("distinct values per label:\n", nunique.to_string())
    if len(chk) > 5 and not (nunique > 1).all():
        raise ValueError(f"constant column(s): {nunique[nunique <= 1].index.tolist()}")
    print(f"\nOK | {len(chk)} rows | total kernel time {(time.time() - started)/3600:.2f} h")
    return out
