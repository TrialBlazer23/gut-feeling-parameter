# MEMORY.md — Living Project Memory

*This file is updated after each completed training run or significant architectural decision. Do not edit manually without a corresponding experiment result or decision record.*

---

## Project Purpose

The Gut Feeling Parameter (GFP) experiment tests whether a curiosity-driven reinforcement learning agent — one that receives no explicit task objective — spontaneously develops sensitivity to sub-threshold precursor signals embedded in a 1-D time-series environment. The core hypothesis is that an agent rewarded for prediction surprise will learn to "protect" against environmental anomalies it has never been directly trained to recognize, purely as a side effect of internal model error dynamics. The precursor signal is designed to be too weak for supervised methods to detect reliably, making the experiment a test of implicit structure learning rather than explicit classification.

---

## Core Mechanism (Config D)

Config D is the primary experimental configuration. Its reward formula is:

```
reward = 0.5 * normalized_prediction_error
         IF (action == 1) AND (pred_error > running_mean + 1.0 * running_std)
         ELSE 0
```

**Gate condition:** The reward is non-zero only when the agent takes the "protect" action (`action == 1`) and the current prediction error exceeds the running mean by at least one standard deviation.

**Running statistics:** `running_mean` and `running_std` are maintained using the Welford online algorithm, which computes exact running statistics without storing the full error history. This is important for memory efficiency in long runs and for numerical stability.

**Normalization:** `normalized_prediction_error` is the raw prediction error divided by a running scale estimate (typically the running standard deviation or a moving average of absolute errors, depending on version).

This formula must not be changed across Config D variants without creating a new named config. See AGENTS.md — Key Invariants.

---

## Architecture Decisions

### LSTM over Transformer
LSTM was chosen as the agent's internal model for signal environments (Phases 1–2) for two reasons:
1. **Temporal locality** — the precursor signal is a short-range temporal pattern. LSTMs with gated memory are well-suited to detecting local temporal structure without the full-sequence attention overhead of Transformers.
2. **Parameter count** — LSTM is substantially smaller for equivalent hidden dimensions, making CPU runs feasible in Phase 1 and keeping Phase 2 GPU runs fast enough for iterative experimentation.

Transformers are noted as a potential upgrade path for Phase 4+ if the experiment scales to longer-range dependencies.

### REINFORCE over PPO
REINFORCE (vanilla policy gradient) was chosen over PPO for Phase 1 because:
- The reward structure is simple enough that on-policy gradients are stable
- PPO's clipping and value network add implementation complexity that obscures the core mechanism in pilot experiments
- PPO is the designated upgrade path for Phase 2+ if training instability is observed

### Welford Online Algorithm for Running Statistics
The Welford algorithm is used instead of storing a full history of prediction errors because:
- It computes exact mean and variance incrementally in O(1) memory
- It avoids numerical issues from summing large arrays
- It matches what a real-time deployed system would use, which keeps Phase 3 (LM application) directly analogous to Phase 1

---

## Phase Status

### Phase 1 (CPU Baseline) — COMPLETE

**Script:** `training/v1_cpu_baseline/cpu_baseline_experiment.py`
**Runtime:** ~5 minutes on CPU
**Configs tested:** A (supervised baseline), B (penalty-only), C (threshold-only), D (curiosity gate)

**Results summary:**

| Config | Precursor Detection Rate | Ward Health | Notes |
|---|---|---|---|
| A | 0.000 | — | Complete failure — supervised baseline cannot detect withheld labels |
| B | 0.290 | low | Ward collapses — false alarm penalty (−0.50) too punishing |
| C | 0.365 | low | Partial detection, ward collapses under penalty pressure |
| D | 0.392 | 100.0 | Best precursor detection; ward health 100.0 is structural (no ward penalty) |

Config D achieved the highest precursor detection rate across all configs (0.392 vs. 0.365 / 0.290 / 0.000). Config A's complete failure confirms that the precursor signal cannot be recovered by a supervised method operating on the same observation space, validating the experimental design.

### Phase 2 (GPU Scale) — WRITTEN, pending run

**Script:** `training/v3_phase2_gpu/phase2_gpu_experiment.py`
**Target runtime:** ~45 minutes on A100 Colab
**Purpose:** Scale Config D training, add Config E (episodic memory variant), statistical comparison across seeds

### Phase 3 (LM Application) — WRITTEN, pending Phase 2 completion

**Script:** `training/v4_phase3_lm/phase3_chaos_core_lm.py`
**Target runtime:** ~2–3 hours on A100 Colab
**Purpose:** Apply the GFP mechanism to a language model (GPT-2) — test whether the curiosity gate improves sensitivity to anomalous token sequences

### Phase 4 — Pending Phase 3 results

Design not finalized. Candidate directions: multi-environment generalization, longer-range precursor signals, comparison against explicit anomaly detection baselines.

---

## Key Observations (Phase 1)

1. **Config D outperforms all structured configs on precursor detection** — detection rate 0.392 vs. 0.365 (Config C), 0.290 (Config B), 0.000 (Config A). The margin over Config C is modest but consistent across seeds.

2. **Config A completely fails** — the supervised baseline cannot detect withheld precursor labels. This is the expected null result and validates that the task is not trivially solvable by direct supervision on the available features.

3. **Config B/C ward collapses** — both configs include a false alarm penalty of −0.50. This penalty is sufficiently punishing relative to the protection reward that the agent learns to suppress protect actions almost entirely, collapsing ward health. This is an important failure mode: explicit penalty structures can suppress the very behavior the experiment is trying to elicit.

4. **Config D's ward health of 100.0 is structural, not behavioral** — Config D has no ward health mechanic (no false alarm penalty, no ward decay). The 100.0 value reflects the absence of a penalty structure, not a learned behavior. This distinction matters for interpreting Phase 2 results if a ward mechanic is re-introduced.

---

## Open Questions

- **Is Config E's memory advantage real or a warm-up effect?** Config E (episodic memory) was designed to give the agent access to past surprise events. If it outperforms Config D, it is unclear whether this is because episodic memory genuinely aids precursor detection or because the memory buffer acts as an implicit warm-up that stabilizes early training.

- **Does Config D's advantage hold across environment variants?** Phase 1 used a single 1-D environment with fixed precursor embedding parameters. It is unknown whether the detection advantage is specific to this environment or generalizes to different signal strengths, noise levels, or precursor durations.

- **Does the mechanism scale to language?** Phase 3 will test whether a GPT-2 model augmented with a curiosity gate shows improved sensitivity to anomalous token sequences. This is the highest-stakes open question for the practical relevance of the GFP mechanism.

---

## User Action Items Pending

1. **Run Phase 2** — open `training/v3_phase2_gpu/phase2_gpu_experiment.py` on A100 Colab (~45 min), save `results/` outputs and download `surprise_traces_phase2.json`
2. **Download Phase 2 outputs** — place JSON results in `results/` and figures in `figures/v3_phase2/`
3. **Run Phase 3** — after reviewing Phase 2 results, run `training/v4_phase3_lm/phase3_chaos_core_lm.py` on A100 Colab (~2–3h)
4. Update this file after each completed run
