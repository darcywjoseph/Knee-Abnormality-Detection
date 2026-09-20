# EXPERIMENT LEDGER

Every run, including failures. If it's not here, it didn't happen.

Format per entry:

```
## EXP-NNN — <short name>
- Date / git SHA / config hash:
- Idea ref: IDEAS.md #N
- Hypothesis (stated BEFORE running):
- Setup (data subset, folds, epochs, seed):
- Result (CV per-label + overall, wall-clock, cost):
- Verdict: CONFIRMED / REFUTED / INCONCLUSIVE
- Backlog updates triggered:
```

---

## EXP-000 — Data profiling (no hypothesis; ground truth gathering)
- Date: 2026-08-08 / branch phase0/competition-spec (uncommitted)
- Setup: downloaded 5 CSVs via Kaggle API; pulled 5 public notebooks; heuristic language pass.
- Facts purchased: 4,407 train studies, 58 gold-labeled (1.3%); test has no Report column;
  metric = mean of 12 AUCs (public notebook, to confirm from rules page); languages
  en 1608 / tr 546 / es 529 / el 321 / nl 280 / de 249 / cyrillic 220 / hr 191 / fr 44 /
  other 419; 4.6% duplicate reports; Fluid_Sensitive==Fat_Suppression degenerate;
  laterality tag absent on ~half; DICOM corpus ~1TB-class (unverified total).
- Verdict: n/a. Backlog updates: killed "identify label list" tasks; added report-hash CV
  grouping; added header-derived weighting; re-ranked LLM extraction to top.

## EXP-001 — LLM (self) report→label extraction, audited on gold
- Date: 2026-08-08 / config: 4 parallel Claude subagents, ~15 reports each, single pass,
  probability outputs, prompt v1 (multilingual, negation/hedge/silence rules).
- Idea ref: IDEAS.md #7
- Hypothesis (stated before run): mean AUC vs gold ≥ 0.85; each OA label ≥ 0.75.
- Setup: all 58 gold-labeled studies' reports; score = per-label ROC AUC of extracted
  probability vs expert label; pooled, no folds (n too small).
- Result: **mean AUC 0.8972**. Per label: ACL .998, MCL .983 (n_pos=9), MedMen .957,
  LatMen .881, MedOA .927, LatOA .847, PFOA .898, Effusion .864, Synovitis .794,
  Baker's .910, Contusion .833, Fracture .874. Wall-clock ~75s, ~100k subagent tokens.
- Verdict: **CONFIRMED** (both criteria). Caveats: n=58 → Wilson CIs wide (±0.05–0.15);
  single-pass; prompt not yet tuned; weakest: Synovitis .794, Contusion .833, Effusion
  .864 (silence-prior handling suspect — reports often omit mild effusion/synovitis).
- Backlog updates: promote EXP-002 full-corpus extraction; add two-pass disagreement →
  per-label confidence weights; add per-language audit; add "silence prior tuned per label
  from gold" idea; rule-extractor idea demoted to fallback.
