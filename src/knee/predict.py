"""Inference: cached studies in, `submission.csv` out.

Loads every complete fold set of checkpoints it can find (several recipes, seeds or the
best-epoch and SWA twins of one run), each built from the training config stored in its
checkpoint, scores every cached study under TTA, averages the members by per-label rank and
writes the submission. The time budget is sized from a probe over the first studies, so a slow
machine drops TTA passes and then members rather than running out of clock.
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
    "InferConfig", "REF_SLOT", "MIN_MARGIN", "checkpoint_roots", "find_checkpoints",
    "model_from_checkpoint", "load_models", "slice_flips", "load_study", "predict", "rank_mean",
    "fit_budget", "write_submission",
]

# Slot -> same-plane reference slot. Some slots are stored back to front relative to their
# reference, which upstream preprocessing cannot always fix.
REF_SLOT = {3: 0, 5: 0, 4: 1}
MIN_MARGIN = 0.02

# Where checkpoints are looked for, in order. On Kaggle every attached dataset or kernel output
# is a child of /kaggle/input (newer images: of /kaggle/input/datasets). The competition mount
# is a child too and is skipped by name: walking its DICOM tree takes about 25 minutes.
CKPT_BASES = (Path("/kaggle/input/datasets"), Path("/kaggle/input"), Path("weights"),
              Path("runs"), Path("."))
SKIP_DIRS = {"competitions", "datasets", "rsna-knee-abnormality-detection"}
# Checkpoint kinds a training run writes, as the prefix of `<kind>_fold<k>.pt`.
CKPT_KINDS = ("best", "swa")

# Values for keys an older checkpoint's training config may lack.
MODEL_DEFAULTS = {
    "N_SLOTS": 6, "ATTN_DIM": 128, "DROPOUT": 0.2, "HEAD": "concat", "HEAD_HIDDEN": 256,
    "SLOT_EMB": False, "POS_EMB": False, "SLICE_STRIDE": 2, "IMG": 320, "MODEL_IMG": None,
    "CACHE_SLICES": 24,
}
# Training-only settings that must be off at inference: no internet, no backward pass.
INFER_OVERRIDES = {"PRETRAINED": False, "GRAD_CHECKPOINT": False}
# The settings that decide how a study is turned into model input; every member must agree.
INPUT_KEYS = ("SLICE_STRIDE", "IMG", "MODEL_IMG", "N_SLOTS")


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
        model_cfg: How a study is turned into model input (`INPUT_KEYS`); `load_models`
            overwrites it with the values shared by every loaded checkpoint.
        n_folds: Fold checkpoints that make up one complete set.
        tta_offsets: Slice-sampling offsets, one forward pass each.
        reserve_s: Seconds held back from the budget for writing the submission.
        budget_h: Wall clock the whole kernel gets.
        device: Torch device string.
        seed: Seed for the random and numpy generators.
    """

    comp_dir: str = "/kaggle/input/rsna-knee-abnormality-detection"
    split: str = "test"
    out_dir: str = "/tmp/cache"
    model_cfg: dict = field(
        default_factory=lambda: {k: MODEL_DEFAULTS[k] for k in INPUT_KEYS})
    n_folds: int = 5
    tta_offsets: tuple[int, ...] = (0, 1)
    reserve_s: int = 900
    budget_h: float = 9.0
    device: str = field(default_factory=_default_device)
    seed: int = 42


def checkpoint_roots(bases) -> list[Path]:
    """Lists the directories whose trees are searched for checkpoints.

    A Kaggle input base is expanded into its children so that the competition mount can be
    left out by name; every other base is searched whole.

    Args:
        bases: Candidate base directories; missing ones are ignored.

    Returns:
        Existing directories to walk.
    """
    roots = []
    for base in bases:
        base = Path(base)
        if not base.is_dir():
            continue
        if base == Path("/kaggle/input") or base.parent == Path("/kaggle/input"):
            roots.extend(c for c in sorted(base.iterdir())
                         if c.is_dir() and c.name not in SKIP_DIRS)
        else:
            roots.append(base)
    return roots


def find_checkpoints(n_folds: int = 5, bases=None) -> list[Path]:
    """Finds every complete fold set of checkpoints and lists their paths, fold-major.

    A set is the `best_fold<k>.pt` or `swa_fold<k>.pt` files of one directory. Only a set
    holding all `n_folds` folds is used, so a half-uploaded weights dataset is reported and
    skipped rather than quietly scored with three folds. Several complete sets, such as two
    recipes, two seeds, or the best-epoch and SWA twins of one run, all become members.

    Args:
        n_folds: How many folds a complete set has.
        bases: Directories to search, or None for `CKPT_BASES`.

    Returns:
        Every set's fold 0, then every set's fold 1, and so on, so that trimming the tail under
        a time budget keeps each set represented.

    Raises:
        SystemExit: No complete set was found, or one set is present twice.
    """
    t0 = time.time()
    sets: dict[tuple[Path, str], dict[int, Path]] = {}
    for root in checkpoint_roots(CKPT_BASES if bases is None else bases):
        for path in root.rglob("*_fold*.pt"):
            kind, _, fold = path.stem.partition("_fold")
            if kind in CKPT_KINDS and fold.isdigit():
                sets.setdefault((path.parent, kind), {})[int(fold)] = path
    complete = {k: v for k, v in sets.items() if set(v) == set(range(n_folds))}
    for (d, kind), v in sorted(sets.items()):
        if (d, kind) not in complete:
            print(f"skipping {d} ({kind}): folds {sorted(v)} of {n_folds}")
    if not complete:
        raise SystemExit(f"no directory holds all {n_folds} folds of any checkpoint kind; "
                         f"searched {[str(b) for b in (CKPT_BASES if bases is None else bases)]}")
    keys = sorted(complete)
    # The same set attached twice (two datasets holding the same run) would count double.
    seen: dict[tuple, Path] = {}
    for key in keys:
        sig = (key[1], tuple(complete[key][f].stat().st_size for f in range(n_folds)))
        if sig in seen:
            raise SystemExit(f"{key[0]} duplicates {seen[sig]} (same kind and file sizes); "
                             "attach each run once")
        seen[sig] = key[0]
    for d, kind in keys:
        print(f"set: {d} ({kind})")
    print(f"checkpoint search: {len(keys)} sets x {n_folds} folds in {time.time() - t0:.1f} s")
    return [complete[key][fold] for fold in range(n_folds) for key in keys]


def model_from_checkpoint(path, device: str):
    """Builds a model from the training config stored in a checkpoint and loads its weights.

    Args:
        path: A checkpoint written by `knee.train` (holds `model` and `cfg`).
        device: Torch device string.

    Returns:
        The model in eval mode and the config it was built from.

    Raises:
        SystemExit: The checkpoint carries no training config.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "cfg" not in ckpt or "model" not in ckpt:
        raise SystemExit(f"{path} carries no training config; cannot rebuild its architecture")
    cfg = dict(MODEL_DEFAULTS)
    cfg.update({k: v for k, v in ckpt["cfg"].items() if v is not None})
    cfg.update(INFER_OVERRIDES)
    model = Model2p5D(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    return model, cfg


def load_models(paths, cfg: InferConfig) -> list[Model2p5D]:
    """Loads one model per checkpoint, each built from its own stored config.

    Every checkpoint must agree on how its input is built; the shared values are written to
    `cfg.model_cfg`, which `load_study` reads.

    Args:
        paths: Checkpoint paths, as returned by `find_checkpoints`.
        cfg: The run's configuration; reads `device`, writes `model_cfg`.

    Returns:
        The loaded models, in the order of `paths`.

    Raises:
        SystemExit: The checkpoints disagree on their input, or one needs upsampled input.
    """
    models, inputs = [], []
    for path in paths:
        model, mcfg = model_from_checkpoint(path, cfg.device)
        models.append(model)
        inputs.append({k: mcfg.get(k) for k in INPUT_KEYS})
        print(f"  {path.parent.name}/{path.name}: {mcfg['BACKBONE']} | head {mcfg['HEAD']}")
    if any(i != inputs[0] for i in inputs):
        raise SystemExit(f"checkpoints disagree on their input settings: {inputs}")
    if inputs[0]["MODEL_IMG"] not in (None, inputs[0]["IMG"]):
        raise SystemExit("a checkpoint was trained on upsampled input (MODEL_IMG), which this "
                         "inference path does not build")
    cfg.model_cfg = dict(inputs[0])
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
