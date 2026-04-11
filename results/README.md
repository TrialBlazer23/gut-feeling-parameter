# results/

This directory stores JSON outputs from completed training runs.

---

## Naming Convention

```
{config}_{phase}_{seed}_{timestamp}.json
```

Examples:
- `config_d_phase1_seed42_20240101T120000Z.json`
- `config_e_phase2_seed7_20240215T083045Z.json`

---

## File Structure

Each results file must contain the following top-level keys:

```json
{
  "config": "Config D",
  "seed": 42,
  "episode_metrics": [
    {"episode": 1, "reward": 0.0, "pred_error": 0.12, "action": 0},
    ...
  ],
  "final_metrics": {
    "precursor_detection_rate": 0.392,
    "ward_health": 100.0,
    "false_alarm_rate": 0.08
  },
  "hyperparams": {
    "lr": 0.001,
    "episodes": 500,
    "hidden_dim": 64,
    "gate_std_multiplier": 1.0
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

`episode_metrics` is a list of per-episode records. `final_metrics` is a flat dict of summary statistics computed over the full run. `hyperparams` mirrors the experiment's `@dataclass` config at run time.

---

## Version Control Policy

Files in this directory are **gitignored by default** because raw results data can be large and changes frequently. Summarized, human-readable results belong in `docs/EXPERIMENT_LOG.md`.

To share or archive results:
- Small files (< 5 MB): commit directly after adding an explicit `!results/your_file.json` override to `.gitignore`
- Large files (5–100 MB): use Git LFS
- Very large files (> 100 MB, e.g., `surprise_traces_phase2.json`): upload to Google Drive or HuggingFace datasets and link from `docs/EXPERIMENT_LOG.md`

---

## How to Populate

Run any training script and it will save results here automatically:

```bash
python training/v1_cpu_baseline/cpu_baseline_experiment.py
```

---

## Current Status

Phase 1 results were captured in session logs during initial development. Structured JSON files will be saved to this directory on the next training run. Phase 2 and Phase 3 results are pending their respective runs (see MEMORY.md for status).
