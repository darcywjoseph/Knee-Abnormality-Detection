# IDEAS — ranked experiment backlog

EV = (est. CV gain × P(works)) ÷ compute cost. Gains in metric points are guesses until the
metric is known (COMPETITION_SPEC §4); ranks will be re-estimated after first baselines.
Cost: S = hours on 1 GPU, M = ~1 GPU-day, L = multi-GPU-days. Status: TODO / RUNNING / DONE /
KILLED (with ledger ref).

## Tier 0 — Infrastructure (do first; EV = enables everything)

| # | Idea | Gain | P | Cost | Notes |
|---|------|------|---|------|-------|
| 1 | Read full rules/data pages, finalize spec | — | 1.0 | S | Blocked: Kaggle access (escalated) |
| 2 | Local metric exactly reproducing competition metric + unit tests | — | 1.0 | S | Blocked on #1 |
| 3 | Grouped+stratified CV (patient groups, site & label-prevalence strata); leakage audit (patient across sites, duplicate exams) | — | 1.0 | S | #1 way to fool ourselves is site leakage |
| 4 | DICOM ingest: sequence auto-ID (sagittal PD-FS vs coronal T1 …), spacing/intensity normalization, laterality canonicalization, cached npy/zarr | — | 1.0 | M | Spec-independent, start now |
| 5 | Submission harness + runtime profiler (≥20% margin vs limit) | — | 1.0 | S | Blocked on #1 for limits |
| 6 | CV↔LB correlation check on ≥3 early submissions | — | 1.0 | S | Gate for trusting CV |

## Tier 1 — Labels from reports (validated differentiator — EXP-001: mean AUC 0.897 vs gold)

| # | Idea | Gain | P | Cost | EV notes |
|---|------|------|---|------|----------|
| 7 | ~~LLM (self) extraction, audited on gold~~ **DONE — EXP-001 CONFIRMED (0.897)** | — | — | — | Weakest: Synovitis .79, Contusion .83, Effusion .86 |
| 7b | **EXP-002: full-corpus extraction (4,349 reports) in tranches; two-pass on subsample → per-(label,study) confidence** | high | 0.9 | S (tokens) | TOP PRIORITY; unblocks all imaging training |
| 7c | Prompt v2: per-label silence-priors tuned on gold; targeted fixes for Synovitis/Contusion/Effusion | med | 0.6 | S | A/B on gold before full run |
| 8 | Per-label extraction confidence → soft labels / loss weights | med | 0.6 | S | Test vs hard labels, one backbone |
| 9 | Fine-tune multilingual encoder (XLM-R / mDeBERTa) on LLM-extracted labels; ensemble with LLM reads | med | 0.5 | S | Distill + disagreement = noise estimate |
| 10 | Per-language & per-site noise-rate estimation; noise-robust loss (GCE/SCE) or co-teaching where noise > threshold | med | 0.5 | M | Only where measured noise is high |
| 11 | Translate all reports to English (LLM) then single-language extractor; compare vs native multilingual | low-med | 0.5 | S | Cheap A/B |
| 12 | Scan Kaggle for public LLM-read label datasets; use as baseline to beat | med | 0.8 | S | Blocked on Kaggle access |
| 13 | Report-section parsing (findings vs impression) before extraction; negation/hedge handling per language | low-med | 0.6 | S | Classic radiology-NLP trap |

## Tier 2 — Imaging model

| # | Idea | Gain | P | Cost | EV notes |
|---|------|------|---|------|----------|
| 14 | 2.5D baseline: slice CNN/ViT encoder + attention pooling per series, concat series → 12 heads | high | 0.9 | M | First real model; the reference point |
| 15 | Backbone head-to-head on fixed fold: DINOv2 vs DINOv3 vs ConvNeXt vs medical-pretrained (e.g. MRI-pretrained ViT) | med | 0.8 | M | Measure, don't assume; check DINOv3 license vs winner-open-source rule first |
| 16 | Sequence-aware fusion: cross-attention across series (sag/cor/ax × T1/T2/PD-FS) with sequence-type embeddings | med | 0.6 | M | After #14 |
| 17 | Full 3D backbone (3D ResNet / SwinUNETR-style) vs 2.5D | med | 0.5 | L | Only if #14 plateaus |
| 18 | MIL over slices with per-finding attention heads (each finding attends its anatomy) | med | 0.5 | M | Interpretable + targeted |
| 19 | Site-robustness: site-balanced sampling, heavy intensity aug, site-adversarial head | med | 0.6 | M | Test-site mix may differ |
| 20 | Laterality flip canonicalization A/B (flip-to-canonical vs flip-aug) | low | 0.7 | S | Cheap, decisive |
| 21 | Per-finding specialist ensembles gated by router vs one multi-head model | low-med | 0.4 | L | Endgame candidate |
| 22 | Knee-region cropping via cheap localizer (drop background slices/FOV) | med | 0.7 | S | Speeds everything; helps efficiency track |

## Tier 3 — Multimodal / creative (≥1 unconventional idea per 10 experiments)

| # | Idea | Gain | P | Cost | EV notes |
|---|------|------|---|------|----------|
| 23 | CLIP-style contrastive pretraining: image encoder ↔ report embeddings (train set only; test has no reports) | med-high | 0.5 | L | Report as training-time teacher |
| 24 | Knowledge distillation: text+image teacher → image-only student | med | 0.5 | M | Pairs with #23 |
| 25 | Auxiliary report-generation head (predict report embedding / key sentences) as regularizer | low-med | 0.4 | M | Unconventional |
| 26 | Anatomy-segmentation pretext task (public knee-MRI seg models, e.g. OAI-trained) → finding-localized features | med | 0.4 | M | Unconventional; check external-data policy |
| 27 | Cross-site pseudo-labeling: train on high-confidence sites, pseudo-label low-confidence-report exams | low-med | 0.4 | M | After noise estimates (#10) |
| 28 | Test-time augmentation (flips/slices sampling) tuned to metric | low | 0.8 | S | Free at endgame |

## Tier 4 — Endgame

| # | Idea | Gain | P | Cost | EV notes |
|---|------|------|---|------|----------|
| 29 | Ensemble across seeds/folds/backbones; optimize blend weights on exact metric | med | 0.9 | S | Standard |
| 30 | Per-label calibration/thresholds if metric rewards it | med | 0.7 | S | Blocked on metric |
| 31 | Efficiency-track candidate: distilled/pruned single model (see #22, #24); decide near deadline | prize | 0.5 | M | Separate metric — verify formula |
| 32 | Final-week reproducibility freeze + 2-submission selection via CV↔LB model | — | 1.0 | S | Non-negotiable |

## Graveyard

(Killed ideas move here with ledger reference and one-line cause of death.)
