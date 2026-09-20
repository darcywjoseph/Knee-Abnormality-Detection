"""DICOM to cache-v3 preprocessing: one compact uint8 .npz per study.

Each study becomes six series slots (sagittal/coronal/axial fat-suppressed, sagittal
non-FS, coronal T1, sagittal T1) of uniformly sampled slices, oriented canonically,
cropped to a fixed physical square and scaled to uint8. Training and inference read the
npz layout documented on `process_study`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import time
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm.auto import tqdm

__all__ = [
    "CacheConfig", "DEFAULT_SLOTS", "SLOT_SPEC", "FALLBACK_ORDER", "HDR_TAGS", "ORDER_TAGS",
    "CANON_SIGN", "to_float", "read_header", "image_centre_x", "classify_contrast",
    "is_localizer", "pick_slots", "study_laterality", "order_files", "sample_indices",
    "read_pixels", "crop_mm", "canonical_flips", "resize", "load_series", "out_path",
    "series_metadata", "build_volume", "process_study", "find_root", "load_study_rows",
    "shard_studies", "pending_studies", "build_cache", "write_manifest", "coverage_report",
]

# Slot order is part of the cache contract: the training head indexes by position.
DEFAULT_SLOTS = ("SAG_FS", "COR_FS", "AX_FS", "SAG_NOFS", "COR_T1", "SAG_T1")

SLOT_SPEC = {
    "SAG_FS":   ("Sagittal", True,  {"PD", "T2", "STIR", "UNK", "GRE"}),
    "COR_FS":   ("Coronal",  True,  {"PD", "T2", "STIR", "UNK", "GRE"}),
    "AX_FS":    ("Axial",    True,  {"PD", "T2", "STIR", "UNK", "GRE"}),
    "SAG_NOFS": ("Sagittal", False, {"PD", "T2", "GRE", "UNK"}),
    "COR_T1":   ("Coronal",  None,  {"T1"}),
    "SAG_T1":   ("Sagittal", None,  {"T1"}),
}

# For an FS slot with no FS series: the next-most fluid-sensitive contrast, in order.
FALLBACK_ORDER = ["T2", "PD", "GRE", "UNK"]

HDR_TAGS = [
    "SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence", "ScanOptions",
    "RepetitionTime", "EchoTime", "InversionTime", "Laterality", "ImageLaterality",
    "PixelSpacing", "Rows", "Columns", "ImageOrientationPatient", "ImagePositionPatient",
    "PhotometricInterpretation", "SeriesInstanceUID",
]
ORDER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"]

CANON_SIGN = {0: 1.0, 1: 1.0, 2: -1.0}   # LPS: +x left, +y posterior, -z inferior (display order)

_RX_LOC = re.compile(r"loc|scout|survey|plan|calib|smartbrain|3.?plane", re.I)
_RX_T1 = re.compile(r"\bt1\b|t1w|t1_|_t1|t1 ", re.I)
_RX_T2 = re.compile(r"\bt2\b|t2w|t2_|_t2|t2 ", re.I)
_RX_PD = re.compile(r"\bpd\b|\bdp\b|proton|dens|pdw", re.I)
_RX_STIR = re.compile(r"stir|tirm", re.I)
_RX_FS = re.compile(r"\bfs\b|fat.?sat|spir|spair|fatsup|\bsat\b|fs_|_fs|fs ", re.I)


@dataclass(frozen=True)
class CacheConfig:
    """Everything the cache build needs beyond the input and output paths.

    Attributes:
        slots: Slot names in cache order; the training head indexes by position.
        n_slices: Slices stored per slot.
        band: (lo, hi) fractions of the ordered stack to sample between.
        img: Output side length in pixels.
        crop_mm: Side of the physical square centre crop, in mm.
        clip_lo: Lower percentile of the per-series clip that maps to 0.
        clip_hi: Upper percentile of the per-series clip that maps to 255.
        min_slices: A series with fewer files is a localizer and never fills a slot.
        lat_dead_mm: |centre x| below this leaves laterality unknown, so nothing is mirrored.
        compress: Whether the npz is written compressed.
        n_shards: How many shards the corpus is split into (1 disables sharding).
        shard_index: Which shard this run builds.
        n_workers: Pool size; 0 means os.cpu_count().
        time_budget_h: Wall-clock hours after which the run stops, resumable.
        seed: Seed for the random and numpy generators.
    """

    slots: tuple[str, ...] = DEFAULT_SLOTS
    n_slices: int = 24
    band: tuple[float, float] = (0.10, 0.90)
    img: int = 320
    crop_mm: float = 140.0
    clip_lo: float = 0.5
    clip_hi: float = 99.5
    min_slices: int = 6
    lat_dead_mm: float = 20.0
    compress: bool = True
    n_shards: int = 3
    shard_index: int = 0
    n_workers: int = 0
    time_budget_h: float = 11.0
    seed: int = 42

    @property
    def n_slots(self) -> int:
        """Number of series slots per study.

        Returns:
            The length of `slots`.
        """
        return len(self.slots)


try:
    import cv2
    cv2.setNumThreads(0)

    def resize(a, size):
        """Resizes one slice to size x size with OpenCV area interpolation.

        Args:
            a: 2-D float array.
            size: Output side length in pixels.

        Returns:
            The resized slice.
        """
        return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
except ImportError:
    from PIL import Image

    def resize(a, size):
        """Resizes one slice to size x size with PIL bilinear interpolation.

        Args:
            a: 2-D float array.
            size: Output side length in pixels.

        Returns:
            The resized slice.
        """
        return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR))


def to_float(x, default=None):
    """Coerces a DICOM value to float.

    Args:
        x: Any value, typically a pydicom element value.
        default: What to return when x is not numeric.

    Returns:
        The float, or default.
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def read_header(series_path):
    """Parses one series' header into a plain dict. Never raises.

    The middle file of the series is read with stop_before_pixels, so one header read
    stands for the whole series.

    Args:
        series_path: Directory holding the series' .dcm files.

    Returns:
        A dict with n_files and hdr_ok always present, and, when the header could be read,
        desc, scan_seq, scan_opts, tr, te, ti, lat_tag, ps_row, ps_col, rows, cols, iop,
        ipp and centre_x. hdr_ok is 0 when only the file count is known.
    """
    files = sorted(p for p in Path(series_path).iterdir() if p.name.lower().endswith(".dcm"))
    h = {"n_files": len(files), "hdr_ok": 0}
    if not files:
        return h
    try:
        ds = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True,
                             force=True, specific_tags=HDR_TAGS)
    except (pydicom.errors.InvalidDicomError, OSError, ValueError, KeyError, AttributeError):
        return h
    h["hdr_ok"] = 1
    h["desc"] = " ".join(str(getattr(ds, t, "") or "") for t in
                         ("SeriesDescription", "ProtocolName", "SequenceName"))
    h["scan_seq"] = str(getattr(ds, "ScanningSequence", "") or "")
    h["scan_opts"] = str(getattr(ds, "ScanOptions", "") or "")
    h["tr"] = to_float(getattr(ds, "RepetitionTime", None))
    h["te"] = to_float(getattr(ds, "EchoTime", None))
    h["ti"] = to_float(getattr(ds, "InversionTime", None))
    lat = str(getattr(ds, "ImageLaterality", "")
              or getattr(ds, "Laterality", "") or "").strip().upper()
    h["lat_tag"] = lat if lat in ("L", "R") else ""
    ps = getattr(ds, "PixelSpacing", None)
    h["ps_row"] = to_float(ps[0]) if ps is not None and len(ps) >= 2 else None
    h["ps_col"] = to_float(ps[1]) if ps is not None and len(ps) >= 2 else None
    h["rows"] = int(getattr(ds, "Rows", 0) or 0)
    h["cols"] = int(getattr(ds, "Columns", 0) or 0)
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    h["iop"] = [float(x) for x in iop[:6]] if iop is not None and len(iop) >= 6 else None
    h["ipp"] = [float(x) for x in ipp[:3]] if ipp is not None and len(ipp) >= 3 else None
    h["centre_x"] = image_centre_x(h)
    return h


def image_centre_x(h):
    """Patient-x (LPS: +x = patient's left) of the image centre, in mm.

    Args:
        h: A header dict from read_header; iop, ipp, ps_row/ps_col and rows/cols are read.

    Returns:
        The x coordinate in mm, or None when the geometry tags are missing.
    """
    if not h.get("iop") or not h.get("ipp") or not h.get("ps_row") or not h.get("rows"):
        return None
    r = np.asarray(h["iop"][:3])   # direction of increasing column index
    c = np.asarray(h["iop"][3:])   # direction of increasing row index
    centre = (np.asarray(h["ipp"]) + r * (h["cols"] - 1) / 2 * h["ps_col"]
              + c * (h["rows"] - 1) / 2 * h["ps_row"])
    return float(centre[0])


def classify_contrast(h, csv_fs):
    """Decides a series' tissue contrast and whether it is fat-suppressed.

    Precedence: SeriesDescription words, then TR/TE (TR < 800 ms -> T1; TE > 60 ms -> T2;
    else PD), with STIR counted as fat-suppressed whatever the other flags say.

    Args:
        h: A header dict from read_header.
        csv_fs: The delivered Fluid_Sensitive flag for this series; truthy forces fs.

    Returns:
        The contrast, one of {T1, PD, T2, STIR, GRE, UNK}, and the fat-suppression flag.
    """
    desc = h.get("desc", "") or ""
    opts = {t.strip().upper() for t in re.split(r"[\\/,;]", h.get("scan_opts") or "")}
    fs = (bool(csv_fs) or bool(_RX_FS.search(desc))
          or bool(opts & {"FS", "FAT_SAT", "FATSAT", "SPIR", "SPAIR"}))
    if _RX_STIR.search(desc) or (h.get("ti") and 80 <= h["ti"] <= 250):
        return "STIR", True
    if _RX_T1.search(desc):
        return "T1", fs
    if _RX_T2.search(desc):
        return "T2", fs
    if _RX_PD.search(desc):
        return "PD", fs
    tr, te = h.get("tr"), h.get("te")
    if tr is not None and te is not None:
        if tr < 800:
            return "T1", fs
        if te > 60:
            return "T2", fs
        return "PD", fs
    if "GR" in (h.get("scan_seq") or ""):
        return "GRE", fs
    return "UNK", fs


def is_localizer(h, min_slices=6):
    """Says whether a series is a localizer/scout rather than a diagnostic acquisition.

    Args:
        h: A header dict from read_header.
        min_slices: Fewest files a diagnostic acquisition may have.

    Returns:
        True when the series has fewer than min_slices files or its description names a
        localizer, survey, plan or calibration scan.
    """
    return h.get("n_files", 0) < min_slices or bool(_RX_LOC.search(h.get("desc", "") or ""))


def pick_slots(meta, slots=DEFAULT_SLOTS):
    """Chooses at most one series per cache slot.

    Pass 1 fills every slot from series that match its spec exactly. Pass 2 fills any FS slot
    still empty with a leftover same-plane non-FS T2, then PD (a minority of studies have no
    fat-suppressed sagittal at all; their T2 is fluid-sensitive and otherwise unused).
    A series never fills two slots. Candidates are ranked by slice count, UID as the
    tie-break, so the result does not depend on the order `meta` arrives in.

    Args:
        meta: One dict per series with uid, plane, fs, contrast, n_files and localizer.
        slots: Slot names in cache order.

    Returns:
        Two lists of one entry per slot, in `slots` order: the chosen series UID per slot
        (None where the slot stays empty), and whether that slot was filled by the pass-2
        fallback.
    """
    used, chosen, fallback = set(), [], [False] * len(slots)
    for name in slots:
        plane, want_fs, contrasts = SLOT_SPEC[name]
        cands = [m for m in meta
                 if m["uid"] not in used and not m["localizer"] and m["plane"] == plane
                 and m["contrast"] in contrasts
                 and (want_fs is None or bool(m["fs"]) == want_fs)]
        cands.sort(key=lambda m: (-m["n_files"], m["uid"]))
        pick = cands[0]["uid"] if cands else None
        if pick is not None:
            used.add(pick)
        chosen.append(pick)
    for i, name in enumerate(slots):
        plane, want_fs, _ = SLOT_SPEC[name]
        if chosen[i] is not None or want_fs is not True:
            continue
        for con in FALLBACK_ORDER:
            cands = [m for m in meta if m["uid"] not in used and not m["localizer"]
                     and m["plane"] == plane and m["contrast"] == con]
            if cands:
                cands.sort(key=lambda m: (-m["n_files"], m["uid"]))
                chosen[i] = cands[0]["uid"]
                used.add(chosen[i])
                fallback[i] = True
                break
    return chosen, fallback


def study_laterality(meta, lat_dead_mm=20.0):
    """Decides which knee a study images.

    The Laterality/ImageLaterality tag wins when any series carries one; otherwise the
    median image-centre x decides, unless it falls inside the dead zone around the midline.

    Args:
        meta: One dict per series, with lat_tag and centre_x.
        lat_dead_mm: Half-width of the dead zone around the midline, in mm.

    Returns:
        The side, one of {'L', 'R', 'U'}, and where it came from: 'tag', 'geom',
        'geom_dead:<x>' or 'none'.
    """
    tags = [m["lat_tag"] for m in meta if m.get("lat_tag")]
    if tags:
        side = max(("L", "R"), key=tags.count)
        return side, "tag"
    xs = [m["centre_x"] for m in meta if m.get("centre_x") is not None]
    if xs:
        x = float(np.median(xs))
        if abs(x) >= lat_dead_mm:
            return ("L" if x > 0 else "R"), "geom"
        return "U", f"geom_dead:{x:.0f}"
    return "U", "none"


def order_files(series_path):
    """Sorts a series' files into through-plane order.

    Each file is keyed by projecting ImagePositionPatient onto the slice normal, falling
    back to InstanceNumber and then to the file's position in the directory listing. When
    fewer than 80 % of files carry usable geometry, the plain name order is kept instead,
    because a part-geometric sort is worse than none.

    Args:
        series_path: Directory holding the series' .dcm files.

    Returns:
        The ordered paths and the slice normal as an unquantised vector, or None when no
        geometry was found.
    """
    files = sorted(p for p in Path(series_path).iterdir() if p.name.lower().endswith(".dcm"))
    if not files:
        return [], None
    keys, n_geom, normal = [], 0, None
    for i, f in enumerate(files):
        k = None
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True,
                                 specific_tags=ORDER_TAGS)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            if ipp is not None and iop is not None and len(ipp) >= 3 and len(iop) >= 6:
                p = np.asarray([float(x) for x in ipp[:3]])
                o = np.asarray([float(x) for x in iop[:6]])
                n = np.cross(o[:3], o[3:])
                k = float(np.dot(p, n))
                if np.isfinite(k):
                    n_geom += 1
                    normal = n
                else:
                    k = None
            if k is None:
                inst = getattr(ds, "InstanceNumber", None)
                if inst is not None:
                    k = float(inst)
        except (pydicom.errors.InvalidDicomError, OSError, ValueError, KeyError,
                AttributeError, TypeError):
            k = None
        keys.append(k if k is not None else float(i))
    if 0 < n_geom < 0.8 * len(files):
        return files, None
    return [f for _, f in sorted(zip(keys, files), key=lambda t: t[0])], normal


def sample_indices(n, k, band):
    """Picks k evenly spaced slice indices from the central band of a stack.

    Args:
        n: How many slices the series has.
        k: How many to sample; indices repeat when k exceeds the band's width.
        band: (lo, hi) fractions of the ordered stack to sample between, inclusive.

    Returns:
        k integer indices, ascending, all within [0, n - 1]. All zeros when n <= 0.
    """
    if n <= 0:
        return np.zeros(k, dtype=int)
    lo, hi = band
    return np.linspace(lo * (n - 1), hi * (n - 1), k).round().astype(int)


def read_pixels(path):
    """Reads one DICOM file's pixels as a rescaled 2-D float array.

    Args:
        path: The .dcm file.

    Returns:
        The slice after RescaleSlope/Intercept, negated when the photometric
        interpretation is MONOCHROME1 so that bright always means high signal, or None
        when the file could not be decoded.
    """
    try:
        ds = pydicom.dcmread(str(path), force=True)
        a = ds.pixel_array.astype(np.float32)
        if a.ndim == 3:
            a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
        a = (a * float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
             + float(getattr(ds, "RescaleIntercept", 0.0) or 0.0))
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = -a
        return a
    # Pixel decoders raise a wide range of types (unsupported transfer syntax, truncated
    # file, missing element). One bad slice must never fail the study, so the list is broad,
    # but it is still narrower than `except Exception`.
    except (pydicom.errors.InvalidDicomError, OSError, ValueError, KeyError, AttributeError,
            TypeError, NotImplementedError, RuntimeError):
        return None


def crop_mm(a, ps_row, ps_col, size_mm):
    """Crops a slice to a fixed physical square around its centre.

    One output pixel is then the same physical size for every series. When the field of
    view is smaller than the requested square, the margin is padded with the slice's own
    minimum value (not zero), so the pad matches the image's background level.

    Args:
        a: The 2-D slice.
        ps_row: Row pixel spacing in mm; a falsy value returns `a` unchanged.
        ps_col: Column pixel spacing in mm; a falsy value returns `a` unchanged.
        size_mm: Side of the square crop in mm.

    Returns:
        A float32 array of round(size_mm / ps_row) x round(size_mm / ps_col) pixels, or
        `a` itself when the pixel spacing is unknown.
    """
    if not ps_row or not ps_col:
        return a
    hr, hc = int(round(size_mm / ps_row)), int(round(size_mm / ps_col))
    n_rows, n_cols = a.shape
    r0, c0 = (n_rows - hr) // 2, (n_cols - hc) // 2
    out = np.full((hr, hc), float(a.min()), dtype=np.float32)
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r0 + hr, n_rows), min(c0 + hc, n_cols)
    out[sr0 - r0: sr1 - r0, sc0 - c0: sc1 - c0] = a[sr0:sr1, sc0:sc1]
    return out


def canonical_flips(iop, normal, side):
    """Lists which axes of (slice, row, col) to flip for canonical orientation.

    Canonical is the standard radiological display: columns run right->left or
    anterior->posterior, rows run superior->inferior or anterior->posterior, and slices
    ascend toward +x/+y/-z. A right knee is then mirrored along whichever axis is aligned
    with patient x, so medial and lateral mean the same image direction for every study.

    Args:
        iop: ImageOrientationPatient as six floats (column direction then row direction),
            or None when the tag is missing.
        normal: The slice normal, or None when no geometry was recovered.
        side: 'L', 'R' or 'U' from study_laterality.

    Returns:
        The axes to flip, a subset of {0: slice, 1: row, 2: col}. Empty when iop is None.
    """
    if iop is None:
        return []
    col_dir = np.asarray(iop[:3])   # increasing column index
    row_dir = np.asarray(iop[3:])   # increasing row index
    dirs = {0: normal, 1: row_dir, 2: col_dir}
    flips = []
    for ax, d in dirs.items():
        if d is None:
            continue
        dom = int(np.argmax(np.abs(d)))
        if d[dom] * CANON_SIGN[dom] < 0:
            flips.append(ax)
        if side == "R" and dom == 0:        # x-aligned axis: mirror right knees
            flips = [f for f in flips if f != ax] if ax in flips else flips + [ax]
    return flips


def load_series(series_path, h, side, n_slices, img, band, crop, clip_lo, clip_hi):
    """Loads one series into the cache's uint8 slot format.

    Slices are ordered, sampled over the central band, physically cropped, resized,
    flipped into canonical orientation and finally scaled by a percentile clip taken over
    the whole sampled stack. A slice that fails to decode is replaced by the previous good
    one, so the slot always has n_slices planes.

    Args:
        series_path: Directory holding the series' .dcm files.
        h: The series' header dict, read for pixel spacing and orientation.
        side: The study's laterality, for the right-knee mirror.
        n_slices: Slices to store.
        img: Output side length in pixels.
        band: (lo, hi) fractions of the stack to sample between.
        crop: Physical crop size in mm.
        clip_lo: Lower percentile of the clip, over the whole sampled stack.
        clip_hi: Upper percentile of the clip.

    Returns:
        A (n_slices, img, img) uint8 array and a note, or (None, note) when the series
        yielded nothing. The note is 'ok', 'partial:<n>' with the count of substituted
        slices, 'no_dcm' or 'all_slices_failed'.
    """
    files, normal = order_files(series_path)
    if not files:
        return None, "no_dcm"
    idx = sample_indices(len(files), n_slices, band)
    planes, bad = [], 0
    for i in idx:
        a = read_pixels(files[int(i)])
        if a is None:
            bad += 1
            planes.append(None)
            continue
        cropped = crop_mm(a, h.get("ps_row"), h.get("ps_col"), crop)
        planes.append(resize(cropped, img).astype(np.float32))
    good = [p for p in planes if p is not None]
    if not good:
        return None, "all_slices_failed"
    last = None
    for j, p in enumerate(planes):
        if p is None:
            planes[j] = last if last is not None else good[0]
        else:
            last = p
    vol = np.stack(planes, axis=0)
    for ax in canonical_flips(h.get("iop"), normal, side):
        vol = np.flip(vol, axis=ax)
    lo, hi = np.percentile(vol, (clip_lo, clip_hi))
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (np.ascontiguousarray((vol * 255.0).round().astype(np.uint8)),
            f"partial:{bad}" if bad else "ok")


def out_path(study, out_dir):
    """Gives the cache file path for one study.

    Args:
        study: StudyInstanceUID.
        out_dir: Directory the cache is written to.

    Returns:
        The .npz path under the output directory.
    """
    return Path(out_dir) / f"{study}.npz"


def series_metadata(study, rows, series_dir, cfg):
    """Reads every series of one study and classifies it.

    Args:
        study: StudyInstanceUID.
        rows: The study's rows from {split}_series.csv, as dicts.
        series_dir: Root of the {split}_series tree.
        cfg: The CacheConfig for this run.

    Returns:
        The per-series dicts slot selection needs (each carrying its header under 'h'),
        and the flat metadata rows written to the series-metadata CSV.
    """
    meta, meta_rows = [], []
    for r in rows:
        uid = str(r["SeriesInstanceUID"])
        path = Path(series_dir) / str(study) / uid
        h = read_header(path) if path.is_dir() else {"n_files": 0, "hdr_ok": 0}
        contrast, fs = classify_contrast(h, r.get("Fluid_Sensitive", 0))
        m = {"uid": uid, "plane": str(r.get("Anatomical_Plane", "")), "fs": fs,
             "contrast": contrast, "n_files": h.get("n_files", 0),
             "localizer": is_localizer(h, cfg.min_slices), "lat_tag": h.get("lat_tag", ""),
             "centre_x": h.get("centre_x"), "h": h}
        meta.append(m)
        meta_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": uid,
                          "plane": m["plane"],
                          "csv_fs": int(r.get("Fluid_Sensitive", 0) or 0),
                          "fs": int(fs), "contrast": contrast,
                          "n_files": m["n_files"], "localizer": int(m["localizer"]),
                          "tr": h.get("tr"), "te": h.get("te"), "lat_tag": m["lat_tag"],
                          "centre_x": m["centre_x"], "ps_row": h.get("ps_row"),
                          "desc": (h.get("desc") or "")[:80]})
    return meta, meta_rows


def build_volume(study, picks, by_uid, series_dir, side, cfg):
    """Loads the chosen series into the study's slot array.

    One unreadable series costs its slot, not the study: the failure is recorded as a note
    and the slot stays zeros with present = 0.

    Args:
        study: StudyInstanceUID.
        picks: The chosen series UID per slot, None where the slot stays empty.
        by_uid: Series metadata dicts keyed by UID, each carrying its header under 'h'.
        series_dir: Root of the {split}_series tree.
        side: The study's laterality, for the right-knee mirror.
        cfg: The CacheConfig for this run.

    Returns:
        The (n_slots, n_slices, img, img) uint8 array, the uint8 presence mask, and the
        per-slot notes as '<slot index>:<note>' strings.
    """
    arr = np.zeros((cfg.n_slots, cfg.n_slices, cfg.img, cfg.img), dtype=np.uint8)
    present = np.zeros(cfg.n_slots, dtype=np.uint8)
    notes = []
    for s, uid in enumerate(picks):
        if uid is None:
            continue
        path = Path(series_dir) / str(study) / uid
        try:
            vol, note = load_series(path, by_uid[uid]["h"], side, cfg.n_slices, cfg.img,
                                    cfg.band, cfg.crop_mm, cfg.clip_lo, cfg.clip_hi)
        # Deliberate isolation point: one unreadable series must cost its slot, not the
        # study. The exception type is recorded in the manifest note.
        except Exception as e:
            vol, note = None, f"err:{type(e).__name__}"
        if vol is None:
            notes.append(f"{s}:{note}")
            continue
        arr[s] = vol
        present[s] = 1
        if note != "ok":
            notes.append(f"{s}:{note}")
    return arr, present, notes


def process_study(study, study_rows, series_dir, out_dir, cfg=CacheConfig()):
    """Builds one study's cache file. Never raises.

    The npz is the contract training and inference read. It holds:
      vol: uint8 (n_slots, n_slices, img, img), the slot volumes in cfg.slots order, zeros
        where the slot is empty;
      present: uint8 (n_slots,), 1 where the slot was filled;
      side: 0-d string array, the study's laterality ('L', 'R' or 'U');
      slot_uids: string (n_slots,), the SeriesInstanceUID behind each slot, '' when empty;
      slot_fallback: uint8 (n_slots,), 1 where an FS slot was filled by a non-FS series.

    Written atomically: the array goes to a .tmp.npz that is renamed into place, so an
    interrupted shard leaves no half-written file for the resume pass to trust.

    Args:
        study: StudyInstanceUID.
        study_rows: Every study's series rows keyed by StudyInstanceUID, as dicts.
        series_dir: Root of the {split}_series tree.
        out_dir: Directory the cache is written to.
        cfg: The CacheConfig for this run.

    Returns:
        A manifest record and one metadata row per series. The record's status is 'ok',
        'skip' when the file already exists, 'empty' when no slot could be filled, or
        'fail' with the traceback in its note.
    """
    dst = out_path(study, out_dir)
    rec = {"StudyInstanceUID": study, "status": "ok", "slots": "", "side": "",
           "side_src": "", "fallback": "", "note": ""}
    meta_rows = []
    try:
        if dst.is_file() and dst.stat().st_size > 0:
            rec["status"] = "skip"
            return rec, meta_rows

        meta, meta_rows = series_metadata(study, study_rows.get(study, []), series_dir, cfg)
        side, side_src = study_laterality(meta, cfg.lat_dead_mm)
        picks, fallback = pick_slots(meta, cfg.slots)
        by_uid = {m["uid"]: m for m in meta}
        arr, present, notes = build_volume(study, picks, by_uid, series_dir, side, cfg)

        if present.sum() == 0:
            rec.update(status="empty", note=";".join(notes))
            return rec, meta_rows

        tmp = dst.with_suffix(".tmp.npz")
        save = np.savez_compressed if cfg.compress else np.savez
        save(tmp, vol=arr, present=present, side=np.array(side),
             slot_uids=np.array([u or "" for u in picks]),
             slot_fallback=np.array(fallback, dtype=np.uint8))
        os.replace(tmp, dst)
        rec.update(slots="".join(str(int(x)) for x in present), side=side, side_src=side_src,
                   fallback="".join(str(int(x)) for x in fallback), note=";".join(notes))
        return rec, meta_rows
    # Deliberate isolation point: this is the whole body of a pool worker, and a shard of
    # 1500 studies must not die on one of them. The traceback goes into the manifest.
    except Exception:
        rec.update(status="fail", note=traceback.format_exc(limit=2).replace("\n", " | "))
        return rec, meta_rows


def find_root(comp_dir, split="train"):
    """Locates the competition mount: a dir holding {split}.csv and {split}_series/.

    Args:
        comp_dir: The configured mount point, tried first; several Kaggle layouts and the
            local data/ directory are tried after it.
        split: 'train' or 'test'.

    Returns:
        The directory that holds both the split CSV and the series tree.

    Raises:
        FileNotFoundError: No candidate directory holds both.
    """
    candidates = [
        Path(comp_dir),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("data"),
        Path("."),
    ]

    def ok(c):
        """Says whether one candidate directory is the competition mount.

        Args:
            c: Candidate directory.

        Returns:
            True when it holds both {split}.csv and {split}_series/.
        """
        return (c / f"{split}.csv").is_file() and (c / f"{split}_series").is_dir()

    for c in candidates:
        if ok(c):
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for d1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for c in [d1] + sorted(p for p in d1.iterdir() if p.is_dir()):
                if ok(c):
                    return c
    raise FileNotFoundError(
        f"competition mount not found (looked at {comp_dir} and the usual alternatives)")


def load_study_rows(root, split="train"):
    """Reads {split}_series.csv and groups its rows by study.

    Args:
        root: The competition mount from find_root.
        split: 'train' or 'test'.

    Returns:
        The series rows as dicts, keyed by StudyInstanceUID.
    """
    series_df = pd.read_csv(Path(root) / f"{split}_series.csv")
    return {s: g.to_dict("records") for s, g in series_df.groupby("StudyInstanceUID")}


def shard_studies(studies, n_shards, shard_index):
    """Keeps the studies belonging to one shard.

    The corpus is built in shards because a notebook's output is capped in size; the split
    is by CRC32 of the UID, so it is stable and needs no shared state.

    Args:
        studies: All StudyInstanceUIDs in scope.
        n_shards: How many shards the corpus is split into; 1 keeps everything.
        shard_index: Which shard to keep.

    Returns:
        The studies of that shard, in the order given.
    """
    if n_shards <= 1:
        return list(studies)
    return [s for s in studies if zlib.crc32(s.encode()) % n_shards == shard_index]


def pending_studies(studies, out_dir):
    """Drops the studies already cached, so a run resumes where it stopped.

    Args:
        studies: StudyInstanceUIDs in scope.
        out_dir: Directory the cache is written to.

    Returns:
        The studies with no non-empty .npz yet.
    """
    return [s for s in studies
            if not (out_path(s, out_dir).is_file() and out_path(s, out_dir).stat().st_size > 0)]


def build_cache(todo, study_rows, series_dir, out_dir, cfg=CacheConfig()):
    """Runs the per-study worker over a shard, under a wall-clock budget.

    Args:
        todo: StudyInstanceUIDs to build.
        study_rows: Every study's series rows keyed by StudyInstanceUID.
        series_dir: Root of the {split}_series tree.
        out_dir: Directory the cache is written to.
        cfg: The CacheConfig for this run; n_workers and time_budget_h are read here.

    Returns:
        The manifest records, the series metadata rows and a count per status. The run
        stops early once the budget is spent; rerunning resumes.
    """
    if not todo:
        return [], [], {}
    n_workers = cfg.n_workers or os.cpu_count() or 2
    budget = cfg.time_budget_h * 3600
    t0 = time.time()
    records, meta_all, counts = [], [], {}
    worker = _StudyWorker(study_rows, str(series_dir), str(out_dir), cfg)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        it = pool.imap_unordered(worker, todo, chunksize=1)
        bar = tqdm(it, total=len(todo), desc="studies", smoothing=0.02)
        for i, (rec, mrows) in enumerate(bar, 1):
            records.append(rec)
            meta_all.extend(mrows)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            elapsed = time.time() - t0
            rate = elapsed / i
            bar.set_postfix(ok=counts.get("ok", 0), bad=len(records) - counts.get("ok", 0),
                            eta_h=f"{rate * (len(todo) - i) / 3600:.2f}")
            if elapsed > budget:
                print(f"\ntime budget reached at {i}/{len(todo)}; rerun to resume.")
                pool.terminate()
                break
    return records, meta_all, counts


@dataclass
class _StudyWorker:
    """Picklable one-argument wrapper around process_study, for the pool.

    Attributes:
        study_rows: Every study's series rows keyed by StudyInstanceUID.
        series_dir: Root of the {split}_series tree.
        out_dir: Directory the cache is written to.
        cfg: The CacheConfig for this run.
    """

    study_rows: dict
    series_dir: str
    out_dir: str
    cfg: CacheConfig = field(default_factory=CacheConfig)

    def __call__(self, study):
        """Builds one study.

        Args:
            study: StudyInstanceUID.

        Returns:
            Whatever process_study returns for it.
        """
        return process_study(study, self.study_rows, self.series_dir, self.out_dir, self.cfg)


def write_manifest(records, meta_rows, out_dir, split="train", shard_index=0):
    """Merges this run's records into the shard's manifest and metadata CSVs.

    Both are written beside the cache directory and merged with what an earlier run left,
    so a resumed shard ends with one row per study and per series.

    Args:
        records: Manifest records from build_cache.
        meta_rows: Series metadata rows from build_cache.
        out_dir: Directory the cache is written to.
        split: 'train' or 'test'.
        shard_index: Which shard this run built; it names the CSVs.

    Returns:
        The merged manifest frame, the merged metadata frame, and the two CSV paths.
    """
    out_dir = Path(out_dir)
    tag = f"{split}_s{shard_index}"
    man_path = out_dir.parent / f"cache_manifest_{tag}.csv"
    meta_path = out_dir.parent / f"series_meta_{tag}.csv"

    new = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["StudyInstanceUID", "status", "slots", "side", "side_src", "fallback", "note"])
    if man_path.is_file():
        new = (pd.concat([pd.read_csv(man_path), new], ignore_index=True)
                 .drop_duplicates("StudyInstanceUID", keep="last"))
    new.to_csv(man_path, index=False)

    meta_df = pd.DataFrame(meta_rows)
    if meta_path.is_file() and len(meta_df):
        meta_df = (pd.concat([pd.read_csv(meta_path), meta_df], ignore_index=True)
                     .drop_duplicates("SeriesInstanceUID", keep="last"))
    if len(meta_df):
        meta_df.to_csv(meta_path, index=False)
    return new, meta_df, (man_path, meta_path)


def coverage_report(manifest_df, meta_df, out_dir, n_in_scope, cfg=CacheConfig()):
    """Summarises what a shard built: size, slot coverage, laterality, failures.

    Args:
        manifest_df: The merged manifest frame from write_manifest.
        meta_df: The merged series metadata frame from write_manifest.
        out_dir: Directory the cache was written to.
        n_in_scope: How many studies the shard covers, for the size projection.
        cfg: The CacheConfig for this run.

    Returns:
        The report as one printable string.
    """
    out_dir = Path(out_dir)
    lines = []
    files = sorted(out_dir.glob("*.npz"))
    size_gb = sum(f.stat().st_size for f in files) / 1e9
    lines.append(f"cache files: {len(files)} | {size_gb:.2f} GB | "
                 f"{size_gb / max(len(files), 1) * 1000:.1f} MB per study")
    if files and len(files) < n_in_scope:
        lines.append(f"projected shard size: {size_gb / len(files) * n_in_scope:.1f} GB")
    lines.append(manifest_df["status"].value_counts().to_string())

    done = manifest_df[manifest_df.status == "ok"]
    if len(done):
        # zfill: a manifest read back on resume has lost the string's leading zeros to the CSV.
        cov = np.array([[int(c) for c in s]
                        for s in done.slots.astype(str).str.zfill(cfg.n_slots)])
        lines.append("\nslot coverage (fraction of studies with the slot filled):")
        for name, c in zip(cfg.slots, cov.mean(axis=0)):
            lines.append(f"  {name:9s} {c:.3f}")
        lines.append(f"\nlaterality: {done.side.value_counts().to_dict()} | source: "
                     f"{done.side_src.str.split(':').str[0].value_counts().to_dict()}")
        fb = np.array([[int(c) for c in s]
                       for s in done.fallback.astype(str).str.zfill(cfg.n_slots)])
        lines.append("fallback-filled fraction per slot: "
                     f"{dict(zip(cfg.slots, fb.mean(axis=0).round(3)))}")
    if len(meta_df):
        lines.append("\ncontrast x fs x plane (all series seen):")
        lines.append(pd.crosstab([meta_df.plane, meta_df.contrast], meta_df.fs).to_string())
        both = meta_df[(meta_df.lat_tag != "") & meta_df.centre_x.notna()]
        if len(both):
            geom = np.where(both.centre_x > 0, "L", "R")
            far = both.centre_x.abs() >= cfg.lat_dead_mm
            agree = (geom[far] == both.lat_tag[far]).mean()
            lines.append(f"\ntag vs geometry agreement (|x| >= dead zone): "
                         f"{agree:.3f} on {far.sum()} series")
    bad = manifest_df[~manifest_df.status.isin(["ok", "skip"])]
    if len(bad):
        lines.append("\nfailures (first 10):")
        lines.append(bad.head(10).to_string(index=False))
    return "\n".join(lines)
