"""Training stage for the 2.5D study classifier: config, loss, schedule, loop and OOF.

Nothing here runs on import. `fit` owns the epoch loop, the two-stage freeze, the time
budget and best-checkpoint saving; `write_oof` and `write_history` put the run's artefacts
next to the checkpoint.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from knee.common import TARGETS

__all__ = [
    "TrainConfig", "seed_all", "amp_scaler", "amp_autocast", "pick_device", "weighted_bce",
    "build_optimizer", "lr_at", "build_scheduler", "per_label_auc", "safe_nanmean",
    "report_auc", "evaluate", "FitResult", "fit", "write_oof", "write_history",
]


@dataclass
class TrainConfig:
    """Every knob of a training run, in the order the notebook lists them.

    Field names are the CFG keys of the Kaggle kernel, so a JSON override written for one
    works for the other.

    Attributes:
        CACHE_KEY: Substring identifying the cache version to index.
        CSV_DIR: Directory holding folds.csv and the weak-label CSV.
        FOLDS_CSV: Fold assignment file name.
        WEAK_CSV: Weak-label file name.
        OUT_DIR: Where the checkpoint, OOF and history are written.
        FOLD: Validation fold.
        N_FOLDS: Number of folds the split was built with.
        BACKBONE: timm model name; "vit" in the name switches on the ViT path.
        PRETRAINED: Load timm pretrained weights.
        VIT_IMG: ViT input side, a multiple of the patch size.
        VIT_UNFREEZE: How many trailing blocks train; >= depth means a full fine-tune.
        VIT_LLRD: Layer-wise LR decay across the unfrozen blocks.
        VIT_LR_BACKBONE: Base LR of the last ViT block.
        VIT_LR_HEAD: Head LR on the ViT path.
        IMG: Cache image side, filled from the cache.
        N_SLOTS: Slots the model sees, filled from the cache and cut by SLOTS.
        CACHE_SLICES: Slices stored per slot, filled from the cache.
        SLOTS: Cache slot indices to keep, or None for all.
        SLICE_DIR_FIX: Path to the slice-direction table, or None.
        SLICE_STRIDE: Read every n-th cache slice.
        ATTN_DIM: Width of the concat head's attention gate.
        HEAD: "concat" or "label_attn".
        HEAD_HIDDEN: label_attn only: width of the attention MLP.
        SLOT_EMB: label_attn only: add a learned per-slot vector to each slice embedding.
        DROPOUT: Head dropout.
        EPOCHS: Epochs to run.
        BATCH_SIZE: Studies per step.
        GRAD_ACCUM: Steps accumulated before an optimizer step.
        GRAD_CHECKPOINT: Recompute backbone activations instead of storing them.
        LR: Trunk LR on the CNN path.
        HEAD_LR_MULT: The fresh head learns this many times faster than the trunk.
        WEIGHT_DECAY: AdamW weight decay.
        WARMUP_FRAC: Fraction of the cosine stage spent warming up.
        STAGE1_EPOCHS: Head-only epochs before the trunk unfreezes; 0 disables the stage.
        GRAD_CLIP: Global grad-norm clip.
        AMP: Mixed precision, CUDA only.
        NUM_WORKERS: DataLoader workers.
        GOLD_WEIGHT: Sample weight of an expert-labelled study.
        CONF_WEIGHT: Weight each label by the extractor's confidence.
        CONF_FLOOR: Smallest confidence weight.
        AUG: Spatial and intensity augmentation on the training split.
        AUG_ROT_DEG: Rotation range in degrees.
        AUG_SCALE: Scale range.
        AUG_SHIFT: Shift range, as a fraction of the image side.
        AUG_GAMMA: Log-gamma range.
        TIME_BUDGET_H: Stop cleanly after this many hours.
        SEED: Seed for python, numpy and torch.
        DEBUG: Cut both splits and run a single epoch.
        DEBUG_STUDIES: Target size of the debug subset.
    """

    CACHE_KEY: str = "rsna-knee-cache-v3"
    CSV_DIR: str = "/kaggle/input/knee-csvs"
    FOLDS_CSV: str = "folds.csv"
    WEAK_CSV: str = "train_weak_labels.csv"
    OUT_DIR: str = "/kaggle/working"

    FOLD: int = 0
    N_FOLDS: int = 5

    BACKBONE: str = "convnext_tiny"
    PRETRAINED: bool = True
    VIT_IMG: int = 224
    VIT_UNFREEZE: int = 6
    VIT_LLRD: float = 0.78
    VIT_LR_BACKBONE: float = 1.5e-5
    VIT_LR_HEAD: float = 1e-3
    IMG: int | None = None
    N_SLOTS: int | None = None
    CACHE_SLICES: int | None = None
    # v3 slot order: 0 sagFS 1 corFS 2 axFS 3 sagNonFS 4 corT1 5 sagT1. The three non-FS
    # slots are present in only 40-63% of studies and zero-filled otherwise, so they can
    # be dropped.
    SLOTS: list[int] | None = None
    SLICE_DIR_FIX: str | None = None
    SLICE_STRIDE: int = 2
    ATTN_DIM: int = 128
    # "concat": shared AttnPool per slot -> concat -> Linear. "label_attn": one attention
    # softmax per label over every slice embedding in the study, so a focal finding on two
    # slices of one slot is not averaged away by a label-agnostic gate.
    HEAD: str = "concat"
    HEAD_HIDDEN: int = 256
    SLOT_EMB: bool = False
    DROPOUT: float = 0.2

    EPOCHS: int = 6
    BATCH_SIZE: int = 2
    GRAD_ACCUM: int = 8
    GRAD_CHECKPOINT: bool = True
    LR: float = 2e-4
    HEAD_LR_MULT: float = 10.0
    WEIGHT_DECAY: float = 0.01
    WARMUP_FRAC: float = 0.05
    STAGE1_EPOCHS: int = 0
    GRAD_CLIP: float = 1.0
    AMP: bool = True
    NUM_WORKERS: int = 2
    GOLD_WEIGHT: float = 3.0
    CONF_WEIGHT: bool = True
    CONF_FLOOR: float = 0.25
    # One random draw per study, so every slot and slice moves together. No flips: the
    # cache normalises laterality and a flip would undo it.
    AUG: bool = True
    AUG_ROT_DEG: float = 7.0
    AUG_SCALE: float = 0.07
    AUG_SHIFT: float = 0.05
    AUG_GAMMA: float = 0.10

    TIME_BUDGET_H: float = 5.2
    SEED: int = 42

    DEBUG: bool = False
    DEBUG_STUDIES: int = 200

    def as_dict(self) -> dict:
        """Returns the config as the plain CFG dict the model and checkpoints expect.

        Returns:
            A dict keyed by field name, in declaration order.
        """
        return dataclasses.asdict(self)

    def apply(self, overrides: dict) -> TrainConfig:
        """Applies a dict of overrides in place, rejecting unknown keys.

        Args:
            overrides: Field name to value.

        Returns:
            This config.

        Raises:
            ValueError: A key is not a config field.
        """
        known = {f.name for f in dataclasses.fields(self)}
        for key, value in overrides.items():
            if key not in known:
                raise ValueError(f"unknown CFG key in TRAIN_OVERRIDES: {key!r}")
            setattr(self, key, value)
        if overrides:
            print("CFG overrides:", overrides)
        if self.DEBUG:
            self.EPOCHS = 1
            print("DEBUG mode: 1 epoch,", self.DEBUG_STUDIES, "studies")
        return self

    @classmethod
    def from_env(cls, overrides: dict | None = None) -> TrainConfig:
        """Builds a config from the defaults, the TRAIN_OVERRIDES env var and a dict.

        Args:
            overrides: Extra overrides, applied after the environment ones.

        Returns:
            The config.
        """
        merged = json.loads(os.environ.get("TRAIN_OVERRIDES", "{}"))
        merged.update(overrides or {})
        return cls().apply(merged)


def seed_all(seed: int) -> None:
    """Seeds python, numpy and torch so a run is reproducible up to cuDNN autotuning.

    Args:
        seed: The seed given to every generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    """Returns the device to train on: CUDA when there is one, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_scaler(enabled: bool):
    """Builds a CUDA gradient scaler.

    Args:
        enabled: False makes every scaler call a no-op.

    Returns:
        A GradScaler for the CUDA device.
    """
    return torch.amp.GradScaler("cuda", enabled=enabled)


def amp_autocast(enabled: bool):
    """Builds the mixed-precision context manager.

    Args:
        enabled: False makes the context a no-op.

    Returns:
        An autocast context manager for the CUDA device.
    """
    return torch.amp.autocast("cuda", enabled=enabled)


def weighted_bce(logits, target, sample_w, label_w=None):
    """BCE on soft targets, weighted per study (gold rows count GOLD_WEIGHT x) and per label.

    The denominator is the batch size, not sample_w.sum(). Dividing by the weight sum
    renormalises within the batch and cancels most of the upweight -- an all-gold batch
    would come out identical to an all-weak one. A fixed denominator makes a gold study
    contribute exactly GOLD_WEIGHT times the gradient of a weak one, which is what the
    config says.

    Args:
        logits: Raw model outputs of shape (B, 12).
        target: Soft targets in [0, 1] of shape (B, 12).
        sample_w: Per-study weight of shape (B,).
        label_w: Per-label weight of shape (B, 12), or None for no label weighting.

    Returns:
        The scalar loss for the batch.
    """
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # (B, 12)
    if label_w is not None:
        loss = loss * label_w
    return (loss.mean(dim=1) * sample_w).sum() / logits.shape[0]


def build_optimizer(cfg: TrainConfig, model) -> tuple[torch.optim.Optimizer, list[bool]]:
    """Builds AdamW with one LR per parameter group.

    The CNN path has two groups, trunk and head. The ViT path gives the head its own LR and
    decays the backbone LR layer-wise: the last block gets VIT_LR_BACKBONE, each earlier one
    VIT_LLRD times less, and the final norm rides with the last block.

    Args:
        cfg: The run config.
        model: The model to optimize.

    Returns:
        The optimizer and, per parameter group, whether it is the head group.
    """
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith(("head", "pool"))]
    if model.is_vit:
        n_blk = len(model.backbone.blocks)
        groups = [{"params": head_params, "lr": cfg.VIT_LR_HEAD}]
        for i, blk in enumerate(model.backbone.blocks):
            ps = [p for p in blk.parameters() if p.requires_grad]
            if ps:
                groups.append({"params": ps,
                               "lr": cfg.VIT_LR_BACKBONE * cfg.VIT_LLRD ** (n_blk - 1 - i)})
        groups.append({"params": [p for p in model.backbone.norm.parameters() if p.requires_grad],
                       "lr": cfg.VIT_LR_BACKBONE})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)
        print("ViT param groups:", [(len(g["params"]), f"{g['lr']:.2e}") for g in groups])
    else:
        trunk_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith(("head", "pool"))]
        optimizer = torch.optim.AdamW(
            [{"params": trunk_params, "lr": cfg.LR},
             {"params": head_params, "lr": cfg.LR * cfg.HEAD_LR_MULT}],
            weight_decay=cfg.WEIGHT_DECAY)
    # Param-group order: ViT = [head, blocks..., norm]; CNN = [trunk, head].
    is_head_group = [g["params"] is head_params
                     or (len(g["params"]) == len(head_params)
                         and g["params"][0] is head_params[0])
                     for g in optimizer.param_groups]
    return optimizer, is_head_group


def lr_at(step: int, warmup: int, cosine_steps: int) -> float:
    """The LR multiplier of the warmup-then-cosine schedule.

    Args:
        step: Step index within the cosine stage, 0-based.
        warmup: Length of the linear warmup, in steps.
        cosine_steps: Length of the whole cosine stage, in steps.

    Returns:
        A multiplier in [0, 1]: linear warmup, then a cosine reaching 0 at cosine_steps.
    """
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(cosine_steps - warmup, 1)
    return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))


def build_scheduler(cfg: TrainConfig, optimizer, is_head_group: list[bool], n_batches: int):
    """Builds the two-stage LambdaLR schedule.

    While the trunk is frozen its multiplier is 0 and only the head moves, on a schedule
    that starts at step 0. The cosine runs over the epochs AFTER the frozen stage and only
    over those, so the LR reaches 0 at the end of training whatever STAGE1_EPOCHS is.

    Args:
        cfg: The run config.
        optimizer: The optimizer to schedule.
        is_head_group: Per parameter group, whether it is the head group.
        n_batches: Batches in one training epoch.

    Returns:
        The scheduler and the total number of optimizer steps.
    """
    steps_per_epoch = max(n_batches // cfg.GRAD_ACCUM, 1)
    stage1_steps = steps_per_epoch * cfg.STAGE1_EPOCHS
    cosine_steps = steps_per_epoch * (cfg.EPOCHS - cfg.STAGE1_EPOCHS)
    total_steps = stage1_steps + cosine_steps
    warmup = max(int(cosine_steps * cfg.WARMUP_FRAC), 1)

    def lr_head(step):
        """The head LR multiplier: the schedule runs from step 0, through both stages.

        Args:
            step: Global optimizer step, 0-based.

        Returns:
            The multiplier applied to the head param group's base LR.
        """
        inner = step if step < stage1_steps else step - stage1_steps
        return lr_at(inner, warmup, cosine_steps)

    def lr_trunk(step):
        """The backbone LR multiplier: zero while the trunk is frozen, then the schedule.

        Args:
            step: Global optimizer step, 0-based.

        Returns:
            The multiplier applied to the trunk param groups' base LRs.
        """
        return 0.0 if step < stage1_steps else lr_at(step - stage1_steps, warmup, cosine_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_head if h else lr_trunk for h in is_head_group])
    if stage1_steps:
        print(f"two-stage: {cfg.STAGE1_EPOCHS} head-only epoch(s) = {stage1_steps} steps, "
              f"then full fine-tune")
    print(f"{steps_per_epoch} optimizer steps/epoch x {cfg.EPOCHS} epochs = {total_steps} "
          f"(warmup {warmup})")
    print(f"LR multiplier at the last step ({total_steps - 1}): "
          f"head {lr_head(total_steps - 1):.4f} | trunk {lr_trunk(total_steps - 1):.4f}")
    return scheduler, total_steps


def per_label_auc(y_true_bin: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Scores each of the 12 labels with ROC AUC.

    `knee.common.per_label_auc` is the DataFrame-shaped version used by the offline
    scoring scripts; it drops unscorable labels instead of padding them, which the training
    history needs, so the two are kept apart.

    Args:
        y_true_bin: Binary labels of shape (n, 12).
        y_pred: Predicted scores of shape (n, 12).

    Returns:
        An array of 12 AUCs, NaN where only one class is present for that label.
    """
    out = []
    for j in range(len(TARGETS)):
        yt = y_true_bin[:, j]
        if len(np.unique(yt)) < 2:
            out.append(np.nan)
        else:
            out.append(roc_auc_score(yt, y_pred[:, j]))
    return np.array(out, dtype=float)


def safe_nanmean(a) -> float:
    """Averages the non-NaN entries.

    Args:
        a: Any array-like of floats.

    Returns:
        The mean of the finite entries, or NaN when every entry is NaN. Unlike
        np.nanmean this raises no RuntimeWarning on an all-NaN input.
    """
    a = np.asarray(a, dtype=float)
    ok = ~np.isnan(a)
    return float(a[ok].mean()) if ok.any() else float("nan")


def report_auc(name: str, aucs: np.ndarray) -> float:
    """Prints the per-label AUCs and their mean.

    Args:
        name: Label for the printed block, e.g. "fold 0 vs GOLD only".
        aucs: The 12 per-label AUCs, NaN where unscorable.

    Returns:
        The mean over the scorable labels, NaN if there are none.
    """
    ok = ~np.isnan(aucs)
    mean_auc = safe_nanmean(aucs)
    print(f"\n{name}: mean AUC = {mean_auc:.4f} over {int(ok.sum())}/12 labels")
    print("  " + "  ".join(f"{t}={a:.3f}" if not np.isnan(a) else f"{t}=--"
                           for t, a in zip(TARGETS, aucs)))
    return mean_auc


@torch.no_grad()
def evaluate(cfg: TrainConfig, model, loader, frame, use_amp: bool = False):
    """Runs the model over a validation split and scores it against weak and gold labels.

    Args:
        cfg: The run config; reads FOLD.
        model: The model to score, moved to its device already.
        loader: DataLoader over the split, in frame order and unshuffled.
        frame: The split's rows, carrying the soft targets and the is_gold flag.
        use_amp: Run the forward pass under autocast.

    Returns:
        A tuple (preds, mean_weak, weak_auc, gold_auc): the (n, 12) probabilities, the mean
        AUC against binarised weak labels, the 12 weak AUCs and the 12 gold AUCs (all NaN
        when the split holds fewer than 4 gold rows).
    """
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for batch in tqdm(loader, desc="valid", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        pres = batch["present"].to(device, non_blocking=True)
        with amp_autocast(use_amp):
            logits = model(x, pres)
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    soft = frame[TARGETS].to_numpy(np.float32)
    weak_auc = per_label_auc((soft > 0.5).astype(int), preds)
    mean_weak = report_auc(f"fold {cfg.FOLD} vs weak labels (>0.5), n={len(frame)}", weak_auc)

    gold_mask = frame["is_gold"].to_numpy() == 1
    gold_auc = np.full(len(TARGETS), np.nan)
    if gold_mask.sum() >= 4:
        gold_auc = per_label_auc((soft[gold_mask] > 0.5).astype(int), preds[gold_mask])
        report_auc(f"fold {cfg.FOLD} vs GOLD only, n={int(gold_mask.sum())}", gold_auc)
    else:
        print(f"\ngold rows in fold {cfg.FOLD}: {int(gold_mask.sum())} -- too few to score")

    return preds, mean_weak, weak_auc, gold_auc


@dataclass
class FitResult:
    """What one training run leaves behind.

    Attributes:
        history: One dict per finished epoch: epoch, train_loss, mean_weak_auc,
            mean_gold_auc.
        best_preds: Validation probabilities of the saved epoch, or None if no epoch ran.
        best_epoch: Which epoch the checkpoint holds, -1 if none.
        best_score: Its mean weak AUC, -inf if no epoch scored.
    """

    history: list[dict] = field(default_factory=list)
    best_preds: np.ndarray | None = None
    best_epoch: int = -1
    best_score: float = float("-inf")


def fit(cfg: TrainConfig, model, train_dl, valid_dl, out_dir, t0: float | None = None):
    """Trains one fold, validating after every epoch and keeping the best checkpoint.

    Stops cleanly mid-epoch once TIME_BUDGET_H is spent, validates the partial epoch and
    returns. Model selection is against the LLM weak labels, so the headline number is a
    training-progress signal, not a leaderboard estimate.

    Args:
        cfg: The run config.
        model: The model, already on its device.
        train_dl: Shuffled training loader.
        valid_dl: Validation loader over a `KneeCache`, in frame order.
        out_dir: Directory for best_fold<FOLD>.pt.
        t0: Start of the time budget; defaults to now.

    Returns:
        A FitResult with the per-epoch history and the best epoch's predictions.
    """
    t0 = time.time() if t0 is None else t0
    out_dir = Path(out_dir)
    device = next(model.parameters()).device
    use_amp = cfg.AMP and device.type == "cuda"
    valid_df = valid_dl.dataset.df

    optimizer, is_head_group = build_optimizer(cfg, model)
    scheduler, _ = build_scheduler(cfg, optimizer, is_head_group, len(train_dl))
    scaler = amp_scaler(use_amp)

    budget = cfg.TIME_BUDGET_H * 3600
    # best_preds must exist even if every epoch scores NaN -- which a tiny DEBUG validation
    # split can genuinely do, when no label has both classes present. Without this the run
    # finishes and writes no checkpoint at all.
    result = FitResult()
    stop = False

    for epoch in range(cfg.EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, seen = 0.0, 0
        bar = tqdm(train_dl, desc=f"epoch {epoch}", smoothing=0.05)

        for i, batch in enumerate(bar):
            x = batch["x"].to(device, non_blocking=True)
            pres = batch["present"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            w = batch["w"].to(device, non_blocking=True)
            wl = batch["wl"].to(device, non_blocking=True)

            with amp_autocast(use_amp):
                loss = weighted_bce(model(x, pres), y, w, wl)

            scaler.scale(loss / cfg.GRAD_ACCUM).backward()

            if (i + 1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += float(loss.detach()) * len(y)
            seen += len(y)
            bar.set_postfix(loss=f"{running / seen:.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                            h=f"{(time.time() - t0) / 3600:.2f}")

            if time.time() - t0 > budget:
                print(f"\ntime budget hit mid-epoch {epoch}; validating and stopping.")
                stop = True
                break

        preds, mean_weak, weak_auc, gold_auc = evaluate(cfg, model, valid_dl, valid_df, use_amp)
        result.history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                               "mean_weak_auc": mean_weak,
                               "mean_gold_auc": safe_nanmean(gold_auc)})
        print(f"epoch {epoch}: loss {running / max(seen, 1):.4f} | "
              f"mean weak AUC {mean_weak:.4f}")

        # NaN never counts as an improvement, but the first epoch is saved regardless so a
        # run whose validation cannot be scored still leaves usable weights behind.
        improved = (not np.isnan(mean_weak)) and mean_weak > result.best_score
        if improved or result.best_preds is None:
            result.best_score = mean_weak if improved else result.best_score
            result.best_epoch = epoch
            torch.save({"model": model.state_dict(), "cfg": cfg.as_dict(), "fold": cfg.FOLD,
                        "epoch": epoch, "mean_weak_auc": mean_weak,
                        "weak_auc": weak_auc.tolist(), "gold_auc": gold_auc.tolist()},
                       out_dir / f"best_fold{cfg.FOLD}.pt")
            result.best_preds = preds
            print(f"  saved best_fold{cfg.FOLD}.pt"
                  + ("" if improved else " (fallback: score is NaN)"))

        gc.collect()
        torch.cuda.empty_cache()
        if stop:
            break

    print(f"\nbest: epoch {result.best_epoch} | mean weak AUC {result.best_score:.4f} | "
          f"{(time.time() - t0) / 3600:.2f} h elapsed")
    print("CAVEAT: model selection and the headline number above are against LLM weak labels, "
          "not gold. Treat them as a training-progress signal only -- they are NOT comparable "
          "to the leaderboard, and they inherit whatever the extractor got wrong.")
    return result


def write_oof(cfg: TrainConfig, result: FitResult, valid_df, out_dir) -> pd.DataFrame:
    """Writes the best epoch's validation predictions as oof_fold<FOLD>.csv.

    Args:
        cfg: The run config; reads FOLD.
        result: What `fit` returned.
        valid_df: The validation rows, in loader order.
        out_dir: Directory to write into.

    Returns:
        The OOF table, empty when the epoch loop never produced predictions.
    """
    out_dir = Path(out_dir)
    if result.best_preds is None:
        # Only reachable if the epoch loop never ran (EPOCHS=0, or the time budget was
        # already spent). Nothing to write, and no reason to raise on the way out.
        print("no validation predictions were produced -- nothing to write.")
        return pd.DataFrame(columns=["StudyInstanceUID"] + TARGETS)
    oof = pd.DataFrame(result.best_preds, columns=TARGETS)
    oof.insert(0, "StudyInstanceUID", valid_df.StudyInstanceUID.to_numpy())
    oof["fold"] = cfg.FOLD
    oof["source"] = valid_df["source"].to_numpy()
    oof.to_csv(out_dir / f"oof_fold{cfg.FOLD}.csv", index=False)
    return oof


def write_history(cfg: TrainConfig, result: FitResult, out_dir) -> None:
    """Writes history_fold<FOLD>.csv and lists what the run produced.

    Args:
        cfg: The run config; reads FOLD.
        result: What `fit` returned.
        out_dir: Directory to write into.
    """
    out_dir = Path(out_dir)
    if result.history:
        hist = pd.DataFrame(result.history)
        hist.to_csv(out_dir / f"history_fold{cfg.FOLD}.csv", index=False)
        print(hist.to_string(index=False))
    print("\nwrote:", [p.name for p in sorted(out_dir.glob(f"*fold{cfg.FOLD}*"))])
