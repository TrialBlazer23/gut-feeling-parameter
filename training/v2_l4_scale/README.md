# v2 — L4 GPU Scale / 5-Notebook Package

## Version
**v2 — L4 GPU Scale / 5-Notebook Package**

## Date
April 10–11, 2026

## Purpose
Systematic scale-up of the v1 pilot. Three goals:

1. Confirm that Config D's precursor detection advantage holds at scale (more seeds, more episodes, longer context windows).
2. Run three targeted sweeps to diagnose the specific failure modes from v1: B/C ward collapse (penalty too aggressive), A's class imbalance (pos_weight), and whether a curriculum can help any config bootstrap detection.
3. Add interpretability tooling to understand _what_ Config D's LSTM is actually representing.

## Hardware
L4 GPU (Google Colab Pro+ recommended). `experiment_core.py` will auto-detect CUDA. CPU fallback is supported for testing but full runs will be prohibitively slow on CPU at this scale.

## Scale
- **Seeds:** 5 (`[42, 7, 13, 99, 1337]`)
- **Configs:** 4 (A, B, C, D)
- **Episodes:** 400 per seed × config
- **Steps per episode:** 1500
- **Hidden size:** 128
- **LSTM layers:** 1

---

## What Was Added

### 5-Notebook Architecture
The experiment is split into five Jupyter notebooks. This separation allows:
- Checkpoints to persist between notebooks (no re-running training when exploring interpretability)
- Targeted re-runs (e.g., re-run only the sweep notebook with new penalty values)
- Easier sharing — each notebook is self-contained for its purpose

All notebooks import from `experiment_core.py`, which is written to disk by `00_setup` and serves as the shared module.

### 3 Targeted Sweeps
Sweep 02 runs three independent sweep families, each addressing a specific v1 finding.

### PCA / Linear Probe Interpretability
Hidden state snapshots collected during training are projected with PCA and probed with a linear classifier (4-class: normal/precursor/threat/other). Measures whether the LSTM's hidden state is geometrically separating the hidden states even though it was never told their names.

### Interactive Dashboard
Notebook 04 provides a drop-down interface (ipywidgets) for exploring per-config, per-seed, per-episode metrics without re-running training.

---

## Notebook Inventory

### `00_setup_and_dependencies.ipynb`
- Verifies GPU availability and prints device info
- Installs any missing packages (`torch`, `numpy`, `matplotlib`, `ipywidgets`, `scikit-learn`)
- Writes `experiment_core.py` to disk (the notebook embeds the full source as a cell)
- Runs a 1-episode smoke test on Config D to confirm the environment and agent work
- Creates the `checkpoints/` and `results/` directories

### `01_core_training_runs.ipynb`
- Loads `experiment_core.py`
- Runs all 4 configs × 5 seeds × 400 episodes
- Saves a checkpoint every 50 episodes (path: `checkpoints/{config}_{seed}_ep{n}.pt`)
- Supports resume: if a checkpoint exists, loads it and continues from that episode
- Logs Welford running stats (mean pred error, std) per step to `results/training_stats_{config}_{seed}.json`
- Produces a per-run summary CSV: `results/v2_main_results.csv`

### `02_large_scale_sweeps.ipynb`
Runs three sweep families. Each sweep trains a Config D (or target config) variant with one hyperparameter varied; all other params held at v2 defaults.

**Sweep 1: False alarm penalty (`false_alarm_penalty` in Config B/C)**
- Range: `[-0.5, -0.3, -0.2, -0.1, -0.05]`
- Target configs: B, C
- Hypothesis: reducing penalty reduces ward collapse while preserving detection
- 2 seeds per point (faster scan)

**Sweep 2: BCE pos_weight (Config A)**
- Range: `[1, 2, 5, 10, 20, 50]`
- Target config: A
- Hypothesis: higher pos_weight pushes Config A toward threat recall; test whether it incidentally improves precursor response
- 2 seeds per point

**Sweep 3: Curriculum (all configs)**
- Interpolates `precursor_mean_shift` from `0.30` down to `0.08` over first 200 episodes
- Hypothesis: starting with an easier detection problem bootstraps the representations before the signal is made difficult
- 2 seeds per config

Results saved to `results/v2_sweep_{sweep_name}.csv`.

### `03_interpretability_and_visualization.ipynb`
- Loads checkpoints from `01_core_training_runs`
- Collects hidden state snapshots (every 10th step, last 5000 steps of each run)
- PCA projection colored by hidden state (normal / precursor / threat) — tests geometric separation
- Linear probe: train a logistic regression on hidden states to classify state; measures accuracy on held-out steps
- Prediction error violin plots: shows pred error distribution split by state; tests whether pred error is highest during precursor
- Writes `results/probe_results.json` and `results/hidden_state_snapshots.npy`

### `04_results_review_dashboard.ipynb`
- Loads all `results/v2_*.csv` and `results/v2_*.json` files
- Interactive ipywidgets dropdowns: select config, seed, metric
- Summary stats panel: mean ± std across seeds for each config
- Sweep comparison plots
- Auto-generated text summary cell (prints a human-readable paragraph summary)

---

## Key Design Decisions

**`experiment_core.py` written to disk by nb00:** All notebooks do `from experiment_core import *`. The source lives in nb00 as a code cell — this makes it easy to diff versions (the cell changes), and means the module is always present on disk after nb00 runs, regardless of how the other notebooks are launched.

**Checkpoint every 50 episodes:** 400 × 1500 steps is ~600k steps per run. If a Colab session crashes, losing more than 50 episodes of progress is painful. 50 is a reasonable checkpoint cadence without excessive disk I/O.

**Welford online stats in the agent:** Instead of storing full prediction error histories and computing stats at episode end, each agent maintains a Welford accumulator. This keeps memory constant across long runs and ensures the surprise threshold adapts as seen in v1.

**5 seeds:** Provides enough replicates for a t-test (df=4) and better characterizes the variance seen in v1's 2-seed run (Config B showed 0.287–0.496 across seeds).

---

## Status
Designed and specified. Notebooks are generated from this spec when running. `experiment_core.py` is complete and importable: `python -c "from experiment_core import *; print('OK')"`

---

## Sweeps Rationale

**Why penalty sweep?** In v1, Configs B and C both show ward collapse despite precursor detection. The default penalty of `-0.5` is aggressive relative to the sparse reward signal at early training. A lower penalty may allow the ward to stabilize while preserving the RL gradient toward detection.

**Why pos_weight sweep?** Config A showed `prec_det=0.000` entirely. Part of this is architectural (it can't detect what it was never labeled). But the class imbalance (threat steps are rare) may also suppress the gradient. Pushing pos_weight higher tests whether A can at least improve threat detection; if not, the architecture explanation is confirmed.

**Why curriculum?** If the precursor signal is hard to detect at `0.08` from the start, agents may converge to ignoring it. Starting at `0.30` (closer to threat magnitude) and annealing down may scaffold early representations before the hard problem phase begins.

---

## Next Version
**v3** — Phase 2 GPU Scale  
`training/v3_phase2_gpu/`

Scale: 5 seeds × 2 configs (D+E) × 200 episodes × 1500 steps, hidden_size=256, A100 GPU.  
Adds Config E (Config D + episodic k-NN memory), Welch t-test + Cohen's d, linear probe (4-class), t-SNE.
