"""Cache discovery, the fold split and the study-level Dataset for the 2.5D trainer.

One item is one study: a (slots, slices, 3, H, W) stack built from the cached volume, its
soft targets and its weights. Every random draw for a study is taken from a Generator
created inside `__getitem__`, because DataLoader seeds torch per worker but not numpy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from knee.common import TARGETS

if TYPE_CHECKING:
    from knee.train import TrainConfig

# cv2 does both the affine augmentation and the ViT resize. Silently skipping them when it is
# missing would train a different model from the one the config describes, so fail at import.
try:
    import cv2
except ImportError as exc:
    raise ImportError("cv2 (opencv-python) is required for augmentation and resizing") from exc
cv2.setNumThreads(0)

__all__ = [
    "find_dir", "find_cache_files", "index_cache", "load_flip_table", "load_table",
    "split_fold", "model_input_side", "random_affine", "KneeCache", "build_datasets",
    "build_loaders", "build_data",
]


def find_dir(configured: str, must_contain: str) -> Path:
    """Finds the directory holding a given file, starting from the configured path.

    Args:
        configured: The directory to try first.
        must_contain: A file name, or a glob pattern when it contains "*".

    Returns:
        The configured directory if it holds must_contain, else the first directory
        under /kaggle/input or data/ that does.

    Raises:
        FileNotFoundError: No directory under the search roots holds must_contain.
    """
    p = Path(configured)
    if p.is_dir() and (any(p.glob(must_contain)) if "*" in must_contain
                       else (p / must_contain).exists()):
        return p
    for base in (Path("/kaggle/input"), Path("data")):
        if not base.is_dir():
            continue
        for root, dirs, _ in os.walk(base):
            # Never descend into the competition mount: it holds ~700k DICOMs and none of
            # them is what we are looking for. Walking it turns a scan into a coffee break.
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            r = Path(root)
            if any(r.glob(must_contain)):
                return r
    raise FileNotFoundError(f"no directory containing {must_contain!r} (tried {configured})")


def find_cache_files(key: str) -> dict[str, Path]:
    """Indexes the cached studies, preferring cache directories whose path contains key.

    Walks the whole input tree (kernel outputs mount at varying depths) but never descends
    into the competition DICOM folders. Falls back to any directory of *.npz if key matches
    none, and to ./data/cache* locally.

    Args:
        key: Substring identifying the wanted cache version, e.g. "rsna-knee-cache-v3".

    Returns:
        A dict mapping StudyInstanceUID to the path of its .npz cache file.

    Raises:
        FileNotFoundError: No *.npz file exists under the search roots.
    """
    bases = [Path("/kaggle/input"), Path("data")]
    hits: dict[str, Path] = {}
    all_hits: dict[str, Path] = {}
    for base in bases:
        if not base.is_dir():
            continue
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            r = Path(root)
            npz = {f[:-4]: r / f for f in fnames if f.endswith(".npz")}
            if not npz:
                continue
            dirs[:] = []                # a cache dir is flat; nothing useful below it
            target = hits if key in str(r) else all_hits
            for k, v in npz.items():
                target.setdefault(k, v)
            print(f"cache dir: {r} ({len(npz)} files)"
                  f"{'' if key in str(r) else '  [key not in path]'}")
    files = hits or all_hits
    if not files:
        raise FileNotFoundError(f"no cache/*.npz found under {bases} (key {key!r})")
    return files


def index_cache(cfg: TrainConfig) -> tuple[dict[str, Path], np.ndarray | None]:
    """Indexes the cache and fills the shape fields of the config from it.

    Slot count, slice count and image side are read from a probe file rather than
    configured, so a cache rebuild cannot silently disagree with the trainer.

    Args:
        cfg: The run config; N_SLOTS, CACHE_SLICES and IMG are written back into it.

    Returns:
        The study-to-path index and the slot subset as an index array, or None for all slots.

    Raises:
        ValueError: SLOTS names a slot the cache does not have.
    """
    cached = find_cache_files(cfg.CACHE_KEY)
    with np.load(next(iter(cached.values()))) as probe:
        cache_shape = probe["vol"].shape
    cfg.N_SLOTS = int(cache_shape[0])
    cfg.CACHE_SLICES = int(cache_shape[1])
    cfg.IMG = int(cache_shape[2])
    print(f"cache: {len(cached)} studies | slots {cfg.N_SLOTS} x slices {cfg.CACHE_SLICES} "
          f"x {cfg.IMG}px")
    if cfg.SLOTS is None:
        return cached, None
    slot_sel = np.asarray(cfg.SLOTS, dtype=int)
    if slot_sel.min() < 0 or slot_sel.max() >= cfg.N_SLOTS:
        raise ValueError(f"SLOTS {cfg.SLOTS} out of range for {cfg.N_SLOTS} cache slots")
    cfg.N_SLOTS = len(slot_sel)
    print(f"slot subset {slot_sel.tolist()} -> {cfg.N_SLOTS} slots")
    return cached, slot_sel


def load_flip_table(path: str | None) -> dict[str, list[int]] | None:
    """Reads the per-(study, slot) slice-direction table.

    Some slots are stored back-to-front relative to their same-plane reference, which makes
    slot k's slice i mean different anatomy study to study. A missing row means "leave
    alone", which is also what a study with an ambiguous correlation margin gets.

    Args:
        path: CSV with StudyInstanceUID and flip<slot> columns, or None for no fix.

    Returns:
        Study to the list of slots to reverse, keeping only studies that flip something,
        or None when path is None.
    """
    if not path:
        return None
    table = pd.read_csv(path).set_index("StudyInstanceUID")
    flip_slots = sorted(int(c[4:]) for c in table.columns if c.startswith("flip"))
    flip = {u: [s for s in flip_slots if r[f"flip{s}"]] for u, r in table.iterrows()}
    flip = {u: v for u, v in flip.items() if v}
    print(f"slice-direction fix: {len(flip)} of {len(table)} studies flip at least one slot")
    return flip


def load_table(cfg: TrainConfig, cached: dict[str, Path]) -> pd.DataFrame:
    """Merges the weak labels with the fold assignment and keeps the cached studies.

    Args:
        cfg: The run config; reads CSV_DIR, FOLDS_CSV and WEAK_CSV.
        cached: The study-to-path index from `index_cache`.

    Returns:
        One row per cached study, carrying the twelve soft targets, fold, source and
        an is_gold flag.
    """
    csv_dir = find_dir(cfg.CSV_DIR, cfg.FOLDS_CSV)
    print("csvs :", csv_dir)
    folds = pd.read_csv(csv_dir / cfg.FOLDS_CSV)
    weak = pd.read_csv(next(csv_dir.rglob(cfg.WEAK_CSV)))

    df = weak.merge(folds, on="StudyInstanceUID", how="inner")
    df["has_cache"] = df.StudyInstanceUID.isin(cached)
    print(f"weak labels {len(weak)} | with fold {len(df)} | cached {int(df.has_cache.sum())}")
    df = df[df.has_cache].reset_index(drop=True)
    df["is_gold"] = (df["source"] == "gold").astype(int)
    print(df.groupby(["fold", "source"]).size().unstack(fill_value=0).to_string())
    return df


def _head_with_gold(d: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Shrinks a split to about k studies while keeping every gold row.

    Args:
        d: The split to shrink, with an is_gold column.
        k: Target number of rows; the gold rows are kept even if they exceed it.
        seed: Shuffle seed.

    Returns:
        The shuffled subset, with a fresh index.
    """
    gold = d[d.is_gold == 1]
    rest = d[d.is_gold == 0].head(max(k - len(gold), 0))
    return pd.concat([gold, rest]).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_fold(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the table into the training and validation halves of one fold.

    Args:
        df: The merged table from `load_table`.
        cfg: The run config; reads FOLD, DEBUG, DEBUG_STUDIES and SEED.

    Returns:
        The train and valid frames, both re-indexed from 0. In DEBUG mode both are cut to
        roughly DEBUG_STUDIES rows, keeping every gold row so the gold AUC block can score.
    """
    train_df = df[df.fold != cfg.FOLD].reset_index(drop=True)
    valid_df = df[df.fold == cfg.FOLD].reset_index(drop=True)
    if cfg.DEBUG:
        n = cfg.DEBUG_STUDIES
        train_df = _head_with_gold(train_df, int(n * 0.8), cfg.SEED)
        valid_df = _head_with_gold(valid_df, int(n * 0.2), cfg.SEED)
    print(f"fold {cfg.FOLD}: train {len(train_df)} ({train_df.is_gold.sum()} gold) | "
          f"valid {len(valid_df)} ({valid_df.is_gold.sum()} gold)")
    return train_df, valid_df


def model_input_side(cfg: TrainConfig) -> int:
    """The image side the backbone is fed.

    Args:
        cfg: The run config; reads BACKBONE, VIT_IMG and IMG.

    Returns:
        VIT_IMG for a ViT backbone, otherwise the cache side.
    """
    return cfg.VIT_IMG if "vit" in cfg.BACKBONE else cfg.IMG


def random_affine(x: np.ndarray, rng: np.random.Generator, cfg: TrainConfig) -> np.ndarray:
    """Applies one random rotation, scale and shift to every image of a study.

    The same transform is used for all slots and slices, so the study stays internally
    aligned. Angles, scale and shift ranges come from AUG_ROT_DEG, AUG_SCALE and AUG_SHIFT.

    Args:
        x: Study volume of shape (slots, S, H, W), float in [0, 1].
        rng: The numpy Generator to draw the transform from.
        cfg: The run config; reads the AUG_* ranges.

    Returns:
        The warped volume, same shape and dtype as x.
    """
    h, w = x.shape[-2:]
    ang = rng.uniform(-cfg.AUG_ROT_DEG, cfg.AUG_ROT_DEG)
    sc = 1.0 + rng.uniform(-cfg.AUG_SCALE, cfg.AUG_SCALE)
    tx, ty = rng.uniform(-cfg.AUG_SHIFT, cfg.AUG_SHIFT, size=2) * np.array([w, h])
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    mat[:, 2] += (tx, ty)
    flat = x.reshape(-1, h, w)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = cv2.warpAffine(flat[i], mat, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(x.shape)


class KneeCache(Dataset):
    """One cached study per item: the 2.5D image stack, its labels and its weights.

    Args:
        frame: Rows of the merged weak-label/fold table to serve.
        train: True to augment and to jitter the slice grid.
        cfg: The run config; reads the AUG_*, SLICE_STRIDE, GOLD_WEIGHT and CONF_* fields.
        cached: The study-to-path index from `index_cache`.
        slot_sel: Slot subset to keep, or None for every slot.
        flip: Slots to reverse per study, from `load_flip_table`, or None.

    Attributes:
        df: The served rows, re-indexed from 0.
        train: Whether augmentation is on.
        y: Soft targets of shape (n, 12).
        w: Per-study sample weight, GOLD_WEIGHT on gold rows.
        wl: Per-label confidence weight of shape (n, 12).
        slice_idx: The cache slice indices read for every slot.
    """

    def __init__(self, frame, train, cfg, cached, slot_sel=None, flip=None):
        self.df = frame.reset_index(drop=True)
        self.train = train
        self.cfg = cfg
        self.cached = cached
        self.slot_sel = slot_sel
        self.flip = flip
        self.img_in = model_input_side(cfg)
        self.slice_idx = np.arange(0, cfg.CACHE_SLICES, cfg.SLICE_STRIDE)
        self.y = frame[TARGETS].to_numpy(np.float32)
        gold = frame["is_gold"].to_numpy() == 1
        self.w = np.where(gold, cfg.GOLD_WEIGHT, 1.0).astype(np.float32)
        # Per-label weight: an LLM score near 0.5 is a hedge or a silence-prior and should
        # pull less than an asserted finding/negation. Gold rows are hard labels: weight 1.
        conf = 2.0 * np.abs(self.y - 0.5)
        wl = (cfg.CONF_FLOOR + (1.0 - cfg.CONF_FLOOR) * conf if cfg.CONF_WEIGHT
              else np.ones_like(self.y))
        wl[gold] = 1.0
        self.wl = wl.astype(np.float32)

    def __len__(self):
        """Returns the number of studies served."""
        return len(self.df)

    def _load(self, study: str) -> tuple[np.ndarray, np.ndarray]:
        """Reads one cached study, applying the slice-direction fix and the slot subset.

        Args:
            study: StudyInstanceUID to read.

        Returns:
            The (slots, CACHE_SLICES, H, W) uint8 volume and the (slots,) float present mask.
        """
        with np.load(self.cached[study]) as d:
            vol = d["vol"]
            present = d["present"].astype(np.float32)
            if self.flip is not None and study in self.flip:
                vol = vol.copy()
                for slot in self.flip[study]:
                    vol[slot] = vol[slot][::-1]
            if self.slot_sel is not None:
                vol, present = vol[self.slot_sel], present[self.slot_sel]
        return vol, present

    def _pick_slices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Chooses the slice grid for one item.

        Args:
            n: Slices available per slot.
            rng: The study's Generator.

        Returns:
            The slice indices, jittered within the stride in train mode so the model does
            not memorise one fixed sampling grid.
        """
        if not self.train:
            return self.slice_idx
        off = rng.integers(0, self.cfg.SLICE_STRIDE)
        return np.clip(self.slice_idx + off, 0, n - 1)

    def _augment(self, vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Applies the spatial and gamma augmentation to a whole study.

        Args:
            vol: The (slots, S, H, W) uint8 volume.
            rng: The study's Generator.

        Returns:
            The augmented volume as float32 back on the 0-255 scale.
        """
        out = random_affine(vol.astype(np.float32) / 255.0, rng, self.cfg)
        gamma = np.float32(np.exp(rng.uniform(-self.cfg.AUG_GAMMA, self.cfg.AUG_GAMMA)))
        out = np.clip(out, 0, 1) ** gamma
        return (out * 255.0).astype(np.float32)

    def _resize(self, x: np.ndarray) -> np.ndarray:
        """Resizes every image of a study to the backbone's input side.

        Args:
            x: Stack of shape (slots, S, 3, H, W).

        Returns:
            The same stack at (slots, S, 3, img_in, img_in).
        """
        k, s_, c, h, w = x.shape
        flat = x.reshape(-1, h, w)
        return np.stack([cv2.resize(f, (self.img_in, self.img_in), interpolation=cv2.INTER_AREA)
                         for f in flat]).reshape(k, s_, c, self.img_in, self.img_in)

    def __getitem__(self, i):
        """Loads, augments and standardises one study.

        Every random draw comes from one freshly seeded Generator created here rather than
        from the global numpy RNG: DataLoader seeds torch per worker but not numpy, so the
        global RNG hands every worker the same jitter and the same intensity scale.

        Args:
            i: Row index into the served frame.

        Returns:
            A dict with "x" (slots, S, 3, H, W) float32, "present" (slots,), "y" (12,),
            "w" (scalar) and "wl" (12,), all torch tensors.
        """
        rng = np.random.default_rng()
        study = self.df.StudyInstanceUID.iloc[i]
        vol, present = self._load(study)

        n = vol.shape[1]
        idx = self._pick_slices(n, rng)
        if self.train and self.cfg.AUG:
            vol = self._augment(vol, rng)

        # 2.5D: channels are the true cache neighbours (i-1, i, i+1), clamped at the ends.
        # These are indexed out of the full stack BEFORE subsampling. Building the triple
        # after striding would give (i-2, i, i+2) at SLICE_STRIDE=2, make each side channel
        # a duplicate of the neighbouring sample's centre channel, and leave the odd cache
        # slices unread whenever the jitter offset is 0.
        x = np.stack([vol[:, np.clip(idx - 1, 0, n - 1)],
                      vol[:, idx],
                      vol[:, np.clip(idx + 1, 0, n - 1)]],
                     axis=2).astype(np.float32) / 255.0   # (slots, S, 3, H, W)

        if self.train:
            x = x * np.float32(rng.uniform(0.9, 1.1)) + np.float32(rng.uniform(-0.05, 0.05))
            x = np.clip(x, 0.0, 1.0)

        if self.img_in != vol.shape[-1]:
            x = self._resize(x)

        # ImageNet-ish standardisation on a greyscale image: one scalar mean/std is enough.
        x = (x - 0.449) / 0.226

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "present": torch.from_numpy(present),
            "y": torch.from_numpy(self.y[i]),
            "w": torch.tensor(self.w[i]),
            "wl": torch.from_numpy(self.wl[i]),
        }


def build_datasets(cfg, train_df, valid_df, cached, slot_sel=None, flip=None):
    """Wraps the two split frames in datasets.

    Args:
        cfg: The run config.
        train_df: Training rows.
        valid_df: Validation rows.
        cached: The study-to-path index.
        slot_sel: Slot subset, or None.
        flip: Slice-direction table, or None.

    Returns:
        The training dataset (augmented) and the validation dataset (not augmented).
    """
    print("model input side:", model_input_side(cfg), "| cache side:", cfg.IMG)
    n_slice = len(np.arange(0, cfg.CACHE_SLICES, cfg.SLICE_STRIDE))
    print(f"{n_slice} slices per slot (stride {cfg.SLICE_STRIDE} over {cfg.CACHE_SLICES})")
    return (KneeCache(train_df, True, cfg, cached, slot_sel, flip),
            KneeCache(valid_df, False, cfg, cached, slot_sel, flip))


def build_loaders(cfg, train_ds, valid_ds, probe: bool = True):
    """Builds the training and validation loaders.

    Args:
        cfg: The run config; reads BATCH_SIZE and NUM_WORKERS.
        train_ds: The training dataset.
        valid_ds: The validation dataset.
        probe: Pull one training batch and print its shapes, which is both a shape check
            and the first use of the shuffle RNG.

    Returns:
        The shuffled, drop_last training loader and the in-order validation loader.
    """
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True,
                          persistent_workers=cfg.NUM_WORKERS > 0)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True,
                          persistent_workers=cfg.NUM_WORKERS > 0)
    if probe:
        sample_batch = next(iter(train_dl))
        print({k: tuple(v.shape) for k, v in sample_batch.items()})
    return train_dl, valid_dl


def build_data(cfg, probe: bool = True):
    """Runs the whole data stage: index the cache, split the fold, build the loaders.

    Args:
        cfg: The run config; its shape fields are filled from the cache.
        probe: Passed to `build_loaders`.

    Returns:
        The training and validation loaders.
    """
    cached, slot_sel = index_cache(cfg)
    flip = load_flip_table(cfg.SLICE_DIR_FIX)
    df = load_table(cfg, cached)
    train_df, valid_df = split_fold(df, cfg)
    train_ds, valid_ds = build_datasets(cfg, train_df, valid_df, cached, slot_sel, flip)
    return build_loaders(cfg, train_ds, valid_ds, probe=probe)
