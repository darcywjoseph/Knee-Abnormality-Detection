# COMPUTE ALLOCATION PLAN — v2 (operator-capped: $100 external compute TOTAL)

> Status: AWAITING EXPLICIT GO-AHEAD before any Lambda API call.
> Operator constraints (2026-08-08): max $100 external compute spend; strict adherence to
> Kaggle compute/platform rules. Lambda key held in chat only — never written to repo,
> memory, or config until approval.

## Strategy under the cap

**Primary compute = Kaggle's free notebook quota** (~30 GPU-h/week on the operator's
single account; sessions up to the platform limit). This is free, rules-native, and the
competition data is already mounted — no 1TB download, no storage rental. All
preprocessing, cache-building, training, and inference happen in Kaggle notebooks:

1. **Cache-builder notebook**: DICOM → compact uint8 slot cache (~20–40 GB), saved as a
   private Kaggle dataset (within Kaggle's private-dataset quota). Rebuilt only when
   preprocessing changes.
2. **Training notebooks**: read the cache dataset, fine-tune encoder, save weights as a
   private dataset (public notebooks demonstrate this attach-weights pattern is normal
   practice; final confirmation pending rules text).
3. **Submission notebook**: offline inference from attached weights, under the time limit.

Weekly quota budget: ~1 cache/preprocess run (~4h) + 3–4 training experiments (~5–6h each)
per week. The experiment loop paces to this; EV ranking decides what gets the hours.

**Kaggle-rules guardrails**: one account (operator's) only — no quota farming via extra
accounts; no private sharing of code/data outside the team; external/off-platform compute
used only if the rules permit it (public precedent says yes; confirm from rules text
before any Lambda training run).

## Lambda reserve — $100 hard cap

Held for the two cases Kaggle quota handles badly, and only after the rules text confirms
off-platform training is allowed:

| Use | Instance | Budget |
|---|---|---|
| R1. Quota overflow in the final 3 weeks (extra folds/seeds for the ensemble) | 1× A100-40GB @ ~$1.29/h | ≤ $60 (~45 h) |
| R2. One-off heavy job that won't fit a Kaggle session (e.g. full-corpus CLIP-teacher pretrain) | 1× A100/H100 burst | ≤ $40 |

- Every Lambda hour is logged per-experiment in experiments/LEDGER.md with $ cost.
- Report to operator at $50 spent and again before crossing $85.
- Instances terminated after each job; no persistent storage rental (re-derive caches on
  Kaggle instead); nothing runs unattended without a ledger entry naming expected wall-clock.
- If the rules forbid external compute for training: Lambda unused, $0 spent, plan still works.

## Label extraction (unchanged, $0)

Agent-run in-session subagent extraction; no GPU, no API budget, not counted against the
$100. Tranches logged in the ledger.
