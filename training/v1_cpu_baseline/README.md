# v1 — CPU Baseline Pilot

## Version
**v1 — CPU Baseline Pilot**

## Date
April 9–10, 2026

## Purpose
First small-scale confirmation run. The goal was to establish whether any of the four training configurations would show evidence of precursor detection in a low-cost CPU environment before committing to larger GPU runs. This was a directional feasibility test, not a conclusive study.

## Hardware
CPU (Google Colab T4 or local). No GPU required. All tensors on CPU throughout.

## Scale
- **Seeds:** 2 (`[42, 7]`)
- **Configs:** 4 (A, B, C, D)
- **Episodes:** 30 per seed × config
- **Steps per episode:** 600
- **Hidden size:** 64
- **LSTM layers:** 1

## What Was Added
First run of all four training configurations in a single self-contained script (`cpu_baseline_experiment.py`). Established the environment signal model (normal / precursor / threat state machine), the LSTM backbone shared across configs, and the four distinct reward/supervision paradigms. No GPU dependencies, no checkpoint logic, no sweeps — just a clean proof-of-concept loop.

---

## Results

### Averaged across 2 seeds

| Config | Description         | ward_health | prec_det | threat_det | fp    |
|--------|---------------------|-------------|----------|------------|-------|
| A      | Supervised          | 86.2        | 0.000    | 0.000      | 0.000 |
| B      | Protection          | 15.6        | 0.365    | 0.357      | 0.342 |
| C      | Prot + Intrinsic    | 22.3        | 0.290    | 0.308      | 0.232 |
| D      | Intrinsic Only      | 100.0       | 0.392    | 0.391      | 0.292 |

### Per-seed breakdown (second 2-seed run)

| Config | Seed | prec_det | ward_health | fp    |
|--------|------|----------|-------------|-------|
| B      | 42   | 0.287    | 0.0         | 0.263 |
| B      | 7    | 0.496    | 5.4         | 0.161 |
| C      | 42   | 0.340    | 0.0         | 0.229 |
| C      | 7    | 0.441    | 4.6         | 0.375 |

---

## Key Observations

**Config A (Supervised BCE):** Complete failure at precursor detection (`prec_det=0.000`). The classifier was deliberately withheld the precursor label — it only saw normal vs. threat. As expected, it learned to classify the two labeled states accurately (high `ward_health` from correct threat response) but developed zero sensitivity to the sub-threshold precursor signals it was never told existed. This is the null baseline: supervised learning on the task as specified cannot solve the hidden detection problem.

**Config D (Intrinsic Only):** Strongest precursor detection (`prec_det=0.392`), perfect `ward_health=100.0`. This is the core finding. Config D receives no objective, no ward health signal, no threat label — only a surprise-gated intrinsic reward when its own prediction error spikes above its running mean. Despite having no extrinsic reason to care about precursors, it develops the highest sensitivity. The ward remains at 100% because the agent never takes false alarm actions (it has no incentive to act, yet it detects).

**Configs B and C (Ward-based):** Both show real precursor detection (higher than Config A), but the ward collapses. The false alarm penalty (`-0.5` for acting when no threat is present) interacts badly with the sparse precursor signal early in training — agents learn to under-act, which partially suppresses detection. The intrinsic bonus in Config C partially counteracts this but doesn't prevent collapse at small scale.

**Seed variance:** Substantial at this scale (30 episodes). Config B shows `prec_det` ranging from 0.287 to 0.496 across seeds. 2 seeds is not enough to characterize distributions reliably.

---

## Limitations
- **Scale is insufficient for convergence.** 30 episodes × 600 steps = 18,000 steps per config × seed. LSTM agents at this scale are still in early learning; the results are directional signals, not stable estimates.
- **Only 2 seeds.** Variance is high and confidence intervals would be wide. The per-seed breakdown shows meaningful spread.
- **No checkpointing or resume.** Each run is start-to-finish; long runs would need to be restarted from scratch.
- **No sweeps.** The false alarm penalty (`-0.5`) and pos_weight for Config A are single fixed values. The ward collapse in B/C may be addressable by tuning.
- **No interpretability.** No hidden state analysis, no probe classifiers, no visualization of what the LSTM representations look like.

---

## Next Version
**v2** — L4 GPU Scale / 5-Notebook Package  
`training/v2_l4_scale/`

Scale: 5 seeds × 4 configs × 400 episodes × 1500 steps, hidden_size=128, L4 GPU.  
Adds: systematic sweeps (penalty, pos_weight, curriculum), checkpoint/resume, PCA + linear probe interpretability, interactive dashboard notebook.
