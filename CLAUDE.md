# CLAUDE.md — Claude Code Instructions

All rules in AGENTS.md apply in full.

---

## When to Use Extended Thinking

Use extended thinking (budget_tokens ≥ 5000) for:

- **Architecture decisions** — evaluating whether to change LSTM to Transformer, adding attention mechanisms, restructuring the reward pipeline
- **Statistical interpretation** — determining whether observed differences between configs are meaningful given sample sizes, selecting appropriate tests, interpreting p-values in context of experiment design
- **Novel mechanism design** — designing new gate conditions, new curiosity formulations, or new ways to structure the precursor signal environment

Do not use extended thinking for routine edits such as adding docstrings, fixing syntax errors, or updating README files.

---

## File Editing

- **Always read an existing file before modifying it.** Do not reconstruct file content from memory.
- **Never change the surprise gate formula** without leaving a comment in the code that explicitly states:
  1. What the formula was before
  2. What it is now
  3. Why the change was made
  
  Example comment pattern:
  ```python
  # FORMULA CHANGE (v3 → v4): Previously gated on pred_error > mean + 1.5*std.
  # Changed to mean + 1.0*std to match Config D invariant defined in AGENTS.md.
  # Reason: v3 threshold was introduced accidentally during GPU port; v1 baseline used 1.0.
  ```

- When modifying a training script, preserve the `@dataclass` config structure. Do not move hyperparameters inline even temporarily.

---

## After Completing Any Experiment Run Analysis

After analyzing results from a completed training run (i.e., after reading a results JSON file and drawing conclusions), update `MEMORY.md`:

1. Move the corresponding phase entry from "WRITTEN — pending run" to "COMPLETE"
2. Add a results summary with exact numbers from the JSON (do not round unless the JSON values are already rounded)
3. Add any new key observations to the "Key Observations" section
4. Update open questions — mark any that are now answered, add new ones that the results raise

---

## On Result Hallucination

**Never hallucinate results.** If a results JSON file is not present in `results/`, or if the file does not contain the requested metric, say so explicitly:

> "No results file found for Phase 2. The experiment has not been run yet, or the output was not saved to results/. Run training/v3_phase2_gpu/phase2_gpu_experiment.py to generate results."

Do not estimate, interpolate, or infer numeric results from partial data unless explicitly asked to do so and clearly labeled as an estimate.

---

## Preferred Approach for New Experiments

1. Create a new version directory under `training/` (e.g., `training/v5_phase4_multienv/`)
2. Copy the nearest prior version's directory as a starting point
3. Make changes to the copy — do not modify prior version scripts in place
4. Add a `README.md` to the new version directory documenting:
   - What changed from the prior version
   - Motivation for the change
   - Expected runtime and compute requirements
   - Which configs are present and what each tests
5. Update `MEMORY.md` to reflect the new phase under "Phase Status"
6. Commit with format: `research(phase<n>): <description>`
