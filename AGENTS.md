# AGENTS.md — Gut Feeling Parameter Research Repository

## Project

**Gut Feeling Parameter (GFP) / Protection Drive**
A curiosity-driven reinforcement learning experiment testing whether an agent with no explicit objective develops sensitivity to sub-threshold precursor signals in a 1-D time-series environment.

This repository is maintained by Necessity Labs (Evan, GitHub: TrialBlazer23).

---

## Purpose

This is a **research repository, not a product**. There is no app, no web frontend, no backend API. Code serves science. The two non-negotiable properties of all work in this repository are:

1. **Completeness** — no stub functions, no placeholder logic, no "TODO: implement later" in any file that is committed to main.
2. **Reproducibility** — every experiment must produce the same result given the same seed. If it doesn't, that is a bug.

---

## Tech Stack

| Package | Minimum Version |
|---|---|
| Python | 3.11+ |
| PyTorch | 2.3+ |
| scikit-learn | 1.4+ |
| transformers (HuggingFace) | 4.40+ |
| scipy | 1.13+ |
| matplotlib | 3.8+ |
| seaborn | 0.13+ |
| numpy | 1.26+ |

---

## Directory Structure

```
gut-feeling-parameter/
├── training/
│   ├── v1_cpu_baseline/        # Phase 1: CPU baseline, 4 configs, short episodes
│   ├── v2_scale/               # Phase 2 early iteration (deprecated, kept for reference)
│   ├── v3_phase2_gpu/          # Phase 2: GPU-scaled Config D/E comparison
│   └── v4_phase3_lm/           # Phase 3: GFP applied to language model (GPT-2)
├── results/                    # JSON outputs from completed training runs
├── figures/                    # Plots organized by version (see figures/README.md)
├── docs/
│   └── EXPERIMENT_LOG.md       # Human-readable log of experiment results and observations
├── AGENTS.md                   # This file — agent instructions
├── CLAUDE.md                   # Claude Code specific instructions
├── MEMORY.md                   # Living project memory — updated after each run
└── STACK.md                    # Technology decisions
```

**Per-directory notes:**

- `training/` — All runnable experiment scripts. Each version subdirectory is self-contained and includes its own README.md documenting changes from the prior version.
- `results/` — Machine-readable outputs only. JSON files saved here by training scripts. Do not commit raw data files over ~10 MB without using Git LFS.
- `figures/` — Generated plots. Subdirectories mirror training version structure. Never commit generated figures produced from truncated/test runs — only from full training runs.
- `docs/` — Human-readable documentation. EXPERIMENT_LOG.md is the canonical record of what has been learned.

---

## Code Standards for Research

### No Stubs
Every function and class committed to main must be fully implemented. Stub implementations with placeholder returns are not acceptable in research code because they silently corrupt result validity.

### Reproducibility — `set_seed` Everywhere
Every training script must call a `set_seed(seed: int)` function at the start of execution that sets:
```python
import random, numpy as np, torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```
The seed used must be logged in the results JSON.

### Results Before Plots
All numeric results must be serialized to a JSON file in `results/` **before** any plot is generated. Plots are derived artifacts. If a training run crashes during plotting, the data must already be safe on disk.

### Hyperparameters in Dataclasses
Hyperparameters must never be hardcoded inline in training logic. They must be defined in a `@dataclass` (or equivalent typed configuration class) at the top of each script. This applies to learning rates, episode counts, hidden dimensions, reward weights, gate thresholds, and all other numeric tuning parameters.

Example pattern:
```python
@dataclass
class ExperimentConfig:
    lr: float = 1e-3
    episodes: int = 500
    hidden_dim: int = 64
    gate_std_multiplier: float = 1.0
    ...
```

---

## How to Run

### Phase 1 — CPU Baseline (~5 minutes, no GPU required)
```bash
python training/v1_cpu_baseline/cpu_baseline_experiment.py
```

### Phase 2 — GPU Scale (A100 Colab, ~45 minutes)
```bash
python training/v3_phase2_gpu/phase2_gpu_experiment.py
```

### Phase 3 — LM Application (A100 Colab, ~2–3 hours)
```bash
python training/v4_phase3_lm/phase3_chaos_core_lm.py
```

Phases 2 and 3 are intended to be run on Google Colab with an A100 runtime. Mount Google Drive before running to ensure results are persisted if the runtime disconnects.

---

## Result Logging Standard

Every training script must save a results JSON file to `results/` upon completion (and at checkpoints for long runs). The JSON must contain the following top-level keys:

```json
{
  "config": "<config name, e.g. 'Config D'>",
  "seed": 42,
  "episode_metrics": [
    {"episode": 1, "reward": 0.0, "pred_error": 0.12, ...},
    ...
  ],
  "final_metrics": {
    "precursor_detection_rate": 0.392,
    "ward_health": 100.0,
    "false_alarm_rate": 0.08,
    ...
  },
  "hyperparams": { ... },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Scripts must not exit without saving results, even on keyboard interrupt. Use a `try/finally` block to guarantee the save.

---

## Commit Format

```
research(<scope>): <description>
```

**Scope** is the affected area: `phase1`, `phase2`, `phase3`, `docs`, `ci`, `config`, etc.

Examples:
```
research(phase2): add Config E episodic memory variant
research(docs): update MEMORY.md with Phase 1 final results
research(ci): add --quick-test flag to cpu_baseline_experiment.py
research(config): bump gate_std_multiplier default to 1.5 — see MEMORY.md
```

---

## Key Invariants

These must not be changed without a version bump and an entry in `docs/EXPERIMENT_LOG.md` explaining the rationale.

### 1. Surprise Gate Formula
The surprise gate formula defines when the agent's curiosity signal is allowed to trigger. Any modification to this formula changes the fundamental experimental condition and invalidates cross-version comparisons. Changes require:
- A new version directory under `training/`
- A comment in the new script explaining what changed and why
- An entry in `docs/EXPERIMENT_LOG.md`

### 2. Config D Core Formula
All Config D variants across all phases must use the following reward formula:

```
reward = 0.5 * normalized_prediction_error
         IF (action == 1) AND (pred_error > running_mean + 1.0 * running_std)
         ELSE 0
```

Where:
- `normalized_prediction_error` is the raw prediction error divided by a running scale estimate
- `running_mean` and `running_std` are maintained via the Welford online algorithm
- The gate threshold multiplier is `1.0` standard deviations (this is the defining parameter of Config D; variants using other multipliers are separate configs)
- `action == 1` is the "protect" action

Deviations from this formula in any Config D file are bugs.
