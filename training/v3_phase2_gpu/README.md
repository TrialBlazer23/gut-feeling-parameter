# v3 — Phase 2 GPU Scale

## Version
**v3 — Phase 2 GPU Scale**

## Date
April 11, 2026 (written; run pending)

## Purpose
Two objectives:

1. **Replication at scale.** Confirm that Config D's precursor detection advantage (first seen in v1, expected to hold in v2) is statistically robust with formal tests (Welch t-test, Cohen's d).
2. **Config E introduction.** Test whether adding episodic k-NN memory to Config D's intrinsic curiosity mechanism improves precursor detection further. If the agent can retrieve previously-seen high-surprise observations as context, it may better recognize recurring precursor patterns.

Phase 2 deliberately narrows to Configs D and E only. Config A is confirmed dead-end; Configs B and C are handled by v2 sweeps. The question at this phase is: how far can the intrinsic-only mechanism go?

## Hardware
A100 GPU (Google Colab Pro+). Expected runtime: ~45 minutes for the full 5-seed × 2-config × 200-episode run. CPU execution is possible for debugging (set `device="cpu"` in the script) but will take hours.

## Scale
- **Seeds:** 5 (`[42, 7, 13, 99, 1337]`)
- **Configs:** 2 (D, E)
- **Episodes:** 200 per seed × config
- **Steps per episode:** 1500
- **Hidden size:** 256
- **LSTM layers:** 1

---

## What's New

### Config E — Config D + Episodic k-NN Memory
Config E extends Config D with a fixed-size episodic memory buffer. At each step:
- The current hidden state `h_t` is compared against all stored (`h_i`, `pred_error_i`) pairs using cosine similarity
- The top-k (k=5) most similar past states are retrieved; their associated prediction errors are averaged to produce a `memory_surprise_bonus`
- The intrinsic reward becomes: `reward = 0.5 * normalized_pred_error + 0.3 * memory_surprise_bonus`
- High-surprise states are pushed into the buffer; the buffer is capped at `memory_size=1000` (FIFO eviction)

The hypothesis is that if the agent has seen a precursor before and found it surprising, it can recognize similar patterns earlier in future episodes.

### Statistical Tests
After collecting all per-episode metrics (last 50 episodes of each 200-episode run per seed), the script runs:
- **Welch's t-test** (two-sided, unequal variance): Config D `prec_det` vs. Config E `prec_det` across the 5-seed sample
- **Cohen's d** (pooled SD): effect size estimate
- Results printed and saved to `results/phase2_stats.json`

### Linear Probe (4-class state classification)
Collects hidden state vectors `h_t` with known ground-truth state labels (normal / precursor / threat / post-threat). Trains a logistic regression probe on 80% of collected vectors and evaluates on held-out 20%. Reports per-class accuracy. Tests whether the LSTM has learned to geometrically separate the hidden states.

**Important:** Use hidden states from the **last 100 episodes** of each run, not last 50. Earlier hidden states may not have converged representations.

### t-SNE Visualization
2-D t-SNE projection of Config D and Config E hidden state vectors, colored by ground-truth state. Tests whether states cluster visibly in the low-dimensional projection. Saved to `plots/tsne_phase2.png`.

**Note:** Use `window_size=100` for t-SNE to ensure enough context. With smaller windows early in training, the clustering may not be visible.

### Surprise Trace Export
Config D and Config E prediction error time series are saved step-by-step to `surprise_traces_phase2.json`. **This file is required by v4** — it is used to seed the Chaos Core training corpus with high-surprise event markers from the signal domain.

Format:
```json
{
  "config_D": {
    "seed_42": [[step, pred_error, state_label], ...],
    ...
  },
  "config_E": { ... }
}
```

---

## Key Output Files

| File | Description | Required by |
|------|-------------|-------------|
| `surprise_traces_phase2.json` | Step-level pred error + state for all D/E runs | v4 (phase3_chaos_core_lm.py) |
| `checkpoints/` | Model weights at episode 50, 100, 150, 200 | v3 analysis |
| `plots/tsne_phase2.png` | t-SNE projection of hidden states | Interpretability |
| `plots/pred_error_violin.png` | Pred error distribution by state | Interpretability |
| `results/phase2_main.csv` | Per-seed per-config per-episode metrics | v3 analysis |
| `results/phase2_stats.json` | t-test, Cohen's d, probe accuracy | Publication |

---

## Status
Script complete at `training/v3_phase2_gpu/phase2_gpu_experiment.py`. Run pending. No v2 results are required as input — v3 runs independently.

---

## Failure Modes and Fixes

These failure modes were identified during script development and are documented here for the first run.

**E doesn't beat D (memory too sparse early in training):**
Extend training to 300 episodes. The k-NN memory buffer needs enough high-surprise events to fill before retrieval becomes useful. At 200 episodes, the buffer may be sparse for the first 50–80 episodes, diluting the memory bonus. Fix: set `n_episodes=300` in the config.

**Linear probe accuracy at chance:**
Use hidden states from the **last 100 episodes**, not last 50. Early in a 200-episode run the representations may not have stabilized. Also verify that all 4 state classes appear in the collected states — short runs may not contain enough threat or precursor steps for a balanced probe.

**t-SNE shows no clustering:**
Increase `window_size` to 100 (currently 50). Also try `perplexity=30` (default) vs. `perplexity=5` (smaller effective neighborhood, may surface local structure). If still no clustering, inspect whether the LSTM is actually updating during training (check gradient norms).

**Surprise trace file is empty or too small for v4:**
The trace collector only records steps where `pred_error > 0` (i.e., after the first step). If the file is small, ensure the seeds are not all starting with identical initial hidden states (confirm `set_seed()` is called before each run, not once globally).

---

## Success Criteria
Any one of the following is sufficient to proceed to v4:
- Config E > Config D on `prec_det` with p < 0.05 (Welch t-test)
- Linear probe accuracy > 50% on precursor class (above chance = 25%)
- Config D `prec_det` > 35% on surprise steps (confirming v1 finding at scale)
- t-SNE shows visible clustering of precursor vs. normal vs. threat regions

---

## Next Version
**v4** — Phase 3 Language Model  
`training/v4_phase3_lm/`

Tests whether the surprise-gating principle scales from synthetic 1-D signals to natural language tasks. Requires `surprise_traces_phase2.json` from this run.
