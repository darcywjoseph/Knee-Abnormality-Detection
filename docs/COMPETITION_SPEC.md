# COMPETITION_SPEC — RSNA Knee Abnormality Detection (Kaggle, 2026)

> Last updated: 2026-08-08 (Kaggle API access obtained; data schemas inspected directly;
> metric and code-requirements sourced from detailed public notebooks — see "Remaining
> unknowns" for what still needs the actual rules text). Re-check weekly.

## 1. Identity & timeline — [VERIFIED via Kaggle API]

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- Research code competition, **$77,000 USD** prizes, **deadline 2026-10-22 23:59 UTC**.
- 539 teams as of 2026-08-08. Operator's account is entered (`userHasEntered: True`).
- Winners announced Nov 2026 (RSNA 2026, Chicago).

## 2. Task & labels — [VERIFIED from data files]

Predict, per **StudyInstanceUID**, probabilities for **12 findings**:

`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture`

- `train.csv`: 4,407 studies × (StudyInstanceUID, Report, 12 label cols).
  **Only 58 studies (1.3%) carry expert labels** (binary 0/1); the other 4,349 have
  labels = NaN and only the free-text `Report`. Deriving labels from reports is the core
  sub-problem, as briefed.
- `test.csv`: StudyInstanceUID only — **no Report column**. Text is unavailable at
  inference; reports are usable only at training time (targets / teacher / weighting).
- Visible test set is a 3-study stub; hidden test swapped in at scoring (code competition).
- Gold-set prevalence (n=58): Effusion 35+, Synovitis 27, Med Men 26, ACL 24, Lat Men 23,
  PF OA 21, Contusion 19, Fracture 18, Med OA 15, Baker's 12, Lat OA 11, **MCL 9** (rarest).

## 3. Evaluation — [VERIFIED via public notebooks; confirm against Evaluation page]

- **Score = unweighted mean of 12 per-label ROC AUCs.**
- Consequences: only per-label rank order matters (no calibration/threshold value);
  ensemble by **averaging ranks**, not probabilities; every label contributes equally, so a
  label at chance forfeits ~(M−0.5)/12 — rare labels deserve *more* attention;
  prevalence is explicitly not guaranteed to match across train/public/private splits.
- **Efficiency track exists** with its own leaderboard and an
  `overview/efficiency-prize-evaluation` page (formula not yet read — typically combines
  score and runtime). A Kaggle-staff notebook publishes the efficiency LB.

## 4. Data details — [VERIFIED from CSVs + public EDA]

- `train_series.csv`/`test_series.csv`: per series `Fluid_Sensitive`, `Fat_Suppression`,
  `Anatomical_Plane` (Sagittal/Coronal/Axial). ~24.4k train series, ~5.5 series/study;
  every study has all 3 planes. **`Fluid_Sensitive == Fat_Suppression` on every row** —
  the two provided flags are degenerate; true contrast weighting (T1/T2/PD) must be
  recovered from DICOM headers (TR/TE/ScanningSequence/SeriesDescription).
- DICOM slices ~1.8MB each, ~150 files/study → full corpus likely **~1TB class** (verify
  before bulk download; plan preprocessing on a big-disk cloud box, train from compact cache).
- **Laterality**: DICOM `Laterality` tag missing on ~half of studies (whole vendors);
  recover side from `ImagePositionPatient`/`ImageOrientationPatient` geometry.
- **Report languages** (heuristic pass, n=4,407): en ~1,608, tr ~546, es ~529, el ~321,
  nl ~280, de ~249, bg/ru (Cyrillic) ~220, hr/sr ~191, fr ~44, other/unresolved ~419.
  Nine-ish languages; Greek reports often use MICRO SIGN (U+00B5) for mu — normalize NFKD.
- **Duplicate/template reports**: ~4.6% of reports are duplicates (4,257 unique of 4,407)
  → CV must group by normalized-report hash as well as by patient/study.
- Report length: median ~977 chars, p95 ~2,452.

## 5. Code requirements — [PARTIAL; from public notebooks, confirm from rules page]

- Kaggle notebook submission, **internet disabled**; attached datasets allowed (pretrained
  weights may be attached as Kaggle datasets — public notebooks do this with DINOv2, and
  train off-platform then attach trained weights, implying this is permitted; CONFIRM).
- Runtime budget: a top public notebook budgets **8.3h against what is presumably a 9h
  limit** (CONFIRM exact limit and whether GPU is P100/T4/L4).

## 6. Known competitive landscape (2026-08-08)

- Public baselines: (a) naive 58-gold-study 2.5D CNN (ignores reports); (b) a very strong
  notebook: multilingual rule-based clause extractor with confidence → weak labels,
  6 anatomically-chosen series slots, DINOv2-small partially unfrozen, per-finding slot
  attention, rank ensembling, report-hash-grouped folds; (c) "DINOsaur v2" variant of same
  lineage (2 seeds × 3 folds, checkpoint soups, TTA). A public "LLM-read labels" table
  exists as an attached Kaggle dataset.
- Public EDA finding: rule/keyword extractors get **recall ≈ 0 on Medial/Lateral/PF OA**
  and mediocre precision on effusion/meniscus (negation) → biggest open edge is
  high-quality LLM extraction, especially for OA labels and non-English languages.
- Gold set n=58 is too small to arbitrate models per-label (MCL: 9 positives; Wilson CIs
  ±10–15pts). Pool predictions across folds before computing AUC; never read single-split
  ±0.02 changes as signal.

## 7. Remaining unknowns (need actual rules/evaluation page text)

1. Exact rules text: external-data policy, pretrained-model policy, open-source obligation
   for winners (affects DINOv3-license question), submission limits/day, team limits.
2. Exact runtime limit + GPU type; efficiency-prize formula.
3. Whether hidden-test prevalence/site mix is disclosed anywhere.
4. Whether the 58 gold labels are meant as validation or trainable (they are in train.csv,
   so trainable — but confirm no rules language restricts).

**Resolution path:** operator pastes Rules + Evaluation + Efficiency-prize page text, or
enables the Claude-in-Chrome extension (claude.ai/chrome) so the agent can read them
directly from the logged-in session.

## Sources

- Kaggle API (`kaggle competitions list/files/download`), data CSVs inspected locally.
- Public notebooks pulled 2026-08-08: pilkwang/rsna-knee-baseline-v1,
  xiaoleilian/rsna26-knee-eda, romantamrazov/rsna-knee-dinosaur-v2,
  debugendless/rsna-knee-abnormality-detection-baseline,
  ryanholbrook/rsna-knee-abnormalities-efficiency-lb (staff).
- Press: RSNA news 2026-08, AuntMinnie 2026-08.
