# Experiment Log — GFP Research

**Repository**: gut-feeling-parameter (Necessity Labs / TrialBlazer23)  
**Log format**: Entries are written at completion of each phase. Pending phases include expected observations based on prior results, not confirmed findings.

---

## Table of Contents

1. [Phase 1 — CPU Baseline Pilot](#phase-1--cpu-baseline-pilot)
2. [Phase 2 — GPU Replication and Extension](#phase-2--gpu-replication-and-extension)
3. [Phase 3 — Language Model Transfer](#phase-3--language-model-transfer)
4. [Phase 4 — TBD](#phase-4--tbd)

---

## Phase 1 — CPU Baseline Pilot

**Status**: Complete  
**Date range**: [To be filled in]  
**Hardware**: CPU (local machine)  
**Script**: `training/v2_phase1_cpu/`

### Scale

| Parameter | Value |
|---|---|
| Seeds per config | 1 |
| Episodes per run | 50 |
| Steps per episode | ~200 (environment-dependent) |
| LSTM hidden size | 32–64 (varies by config) |
| Configs tested | A, B, C, D |
| Config E | Not tested in Phase 1 |

### What Was Tested

Phase 1 established a CPU-scale baseline across four experimental configurations in a synthetic signal environment:

- **Config A** — Supervised binary classification. The agent receives labels for "threat" vs. "normal" states. Precursor labels are withheld. This is the baseline against which all other configs are compared.
- **Config B** — REINFORCE with a ward health reward. The agent protects a simulated ward; false alarms drain ward health. No intrinsic motivation.
- **Config C** — Ward health reward plus a prediction-error bonus (0.5 × normalized prediction error). Tests whether combining extrinsic protection objective with intrinsic motivation helps.
- **Config D** — Pure curiosity. No ward, no objective. Reward = 0.5 × normalized prediction error, gated by action == 1 AND prediction error > mean + 1σ (Welford running stats).

The signal environment generates a time-series with three state types: normal (low amplitude, high regularity), precursor (structured deviation preceding a high-intensity event), and event (high-intensity, brief). Precursor states are the target of detection. No config is given explicit precursor labels.

### What Was Observed

| Config | Precursor Detection Rate | Ward Health (final) | Episode Return |
|---|---|---|---|
| A — Supervised Baseline | 0% | N/A | N/A (loss-based) |
| B — Protection Only | ~36% | ~0 | Low |
| C — Protection + Intrinsic | ~29% | Partial | Moderate |
| D — Intrinsic Only | ~39% | 100 (no ward) | Highest |

Specific observations:

- **Config A** failed to detect any precursor states. This is consistent with the class imbalance problem: precursor timesteps are a small fraction of total timesteps, and without explicit precursor labels, BCE training converges to classifying everything as normal.
- **Config B** achieved ~36% precursor detection but at the cost of near-total ward health collapse. The false alarm penalty appears to have suppressed action == 1 broadly, but precursor states' distinctive signal apparently still triggered some correct detections. The ward collapsing to near-zero suggests the agent was either too aggressive (many false alarms draining health) or too passive (missing actual events). Both pathways are consistent with the observed outcome; they are not distinguishable from Phase 1 data alone.
- **Config C** showed *lower* precursor detection than Config B (~29% vs ~36%), despite having an additional intrinsic motivation component. This is counter-intuitive but consistent with the interpretation that the false alarm penalty in the ward health framework is sufficiently strong to suppress the intrinsic signal's effect. The intrinsic bonus may push the agent to take action == 1 more often during prediction-error moments, but if this increases false alarms against the ward, the net policy gradient is suppressive.
- **Config D** produced the highest precursor detection rate (~39%) with no ward at all. Ward health is trivially 100 (there is no ward to drain). This is the most interpretable result: the agent has no competing signal, and the surprise gate fires disproportionately on precursor states.

### Interpretation

The Phase 1 results are consistent with the core GFP hypothesis: surprise-gated intrinsic reward may provide a latent signal that aligns with precursor detection without explicit supervision. The observation that Config D outperforms Configs B and C is particularly notable because it suggests that adding an extrinsic objective does not help and may actively interfere with the mechanism.

**However**: these conclusions should be held loosely. Each configuration was run once. The differences between 39%, 36%, and 29% detection rates in a single 50-episode run are well within the range of random variation. The supervised baseline (Config A, 0%) is the most robust reference point — it is structurally predictable that a BCE classifier with no precursor labels fails at precursor detection. The comparison between B, C, and D requires replication.

The mechanism story — that prediction error elevates on precursor states because they break the agent's learned regularity model — is plausible but not yet confirmed. It is equally possible that the agent is detecting some other correlated feature of precursor states (e.g., raw amplitude, temporal position in episode) that happens to align with prediction error. The linear probe analysis in Phase 2 is designed to test this.

### Open Questions Raised

1. Are the B/C/D detection rate differences stable across seeds, or are they noise at this scale?
2. What is the precision-recall trade-off for Config D? A recall of 39% with very low precision would be a much weaker finding than 39% recall with acceptable precision.
3. What is the false alarm rate for Config D? (No ward means no cost for false alarms — so Config D's freedom from the ward penalty may be what's allowing it to take more risks, not that it has learned anything structural.)
4. Does the LSTM hidden state of Config D contain a linear projection that separates precursor from normal states? If not, the detection may be superficial.
5. Does Config D's performance degrade with longer episodes or more diverse signal statistics?

---

## Phase 2 — GPU Replication and Extension

**Status**: Pending (script written, not yet run)  
**Scheduled hardware**: Google Colab A100  
**Script**: `training/v3_phase2_gpu/phase2_gpu_experiment.py`

### Scale

| Parameter | Value |
|---|---|
| Seeds per config | 5 |
| Episodes per run | 200 |
| Steps per episode | ~200 |
| LSTM hidden size | 256 |
| Configs tested | A, B, C, D, E |
| Config E | D + episodic k-NN memory (Phase 2 addition) |

### What Will Be Tested

Phase 2 replicates Phase 1 at scale and adds several components:

1. **Multi-seed replication** (5 seeds): Enables statistical testing. The primary analysis is Welch's t-test comparing Config D precursor detection rates against Configs A, B, and C across seeds. Cohen's d will quantify effect size.
2. **Config E**: Adds episodic k-NN memory to Config D. At each step, the agent stores its current hidden state and prediction error in an episodic buffer. Before acting, it retrieves the k nearest neighbors (cosine similarity) and receives a gated signal from the memory. Hypothesis: memory retrieval reinforces precursor recognition by retrieving states similar to past high-error moments.
3. **Linear probe**: A two-layer MLP is trained on LSTM hidden states of Config D (frozen) to predict precursor/normal/event labels. If the probe achieves significantly above-chance accuracy, the agent has implicitly encoded precursor identity.
4. **t-SNE visualization**: LSTM hidden states are projected with t-SNE and colored by state type. A clear separation of precursor states from normal states (without any supervision) would visually support H2.
5. **Precision-recall reporting**: Phase 2 reports both precision and recall for all configs, and includes a random-action baseline and (if warranted) a signal-amplitude threshold baseline.

### Expected Observations (Based on Phase 1)

These are motivated predictions, not confirmed results. Phase 2 may refute them.

- Config D is expected to show statistically significant precursor detection improvement over Config A (Welch t-test p < 0.05), given Phase 1's 39% vs 0% gap.
- The direction of B > D or D > B may reverse or become inconclusive when error bars are added; the ~3 percentage point advantage of D over B in Phase 1 is unlikely to be significant on its own.
- Config E is expected to show equal or higher detection than Config D if the episodic memory signal helps, but the benefit is uncertain — this is the primary exploratory question of Phase 2.
- The linear probe is expected to show above-chance precursor accuracy for Config D, consistent with H2. However, this result may be confounded by amplitude (see Confounds in HYPOTHESIS.md) and must be compared against a probe on raw signal values.

### Open Questions for Phase 2

1. Does Config E add meaningful value over Config D, or does the memory mechanism add noise at this episode length/environment scale?
2. At what training step does Config D's surprise gate begin to consistently fire on precursor states? (Implies a phase transition in the learning dynamics that would be interesting to characterize.)
3. How sensitive is Config D to the Welford threshold multiplier? (Candidate ablation: {0.5, 1.0, 1.5, 2.0} × σ.)
4. Does the linear probe accuracy plateau or continue to improve with more episodes? (Implies whether the representation is still developing at 200 episodes.)

---

## Phase 3 — Language Model Transfer

**Status**: Pending (script written, not yet run)  
**Scheduled hardware**: Google Colab A100  
**Script**: `training/v4_phase3_lm/phase3_chaos_core_lm.py`  
**Prerequisite**: Phase 2 results should be reviewed before running Phase 3. Phase 3 is only well-motivated if Phase 2 provides statistically significant support for H1 and at least suggestive support for H2.

### What Will Be Tested

Phase 3 applies the GFP mechanism to GPT-2 fine-tuning, motivated by the hypothesis that surprise-gated training may improve performance on tasks requiring attention to early contextual signals — specifically causal and temporal reasoning tasks in BIG-Bench Hard.

Three conditions:

- **Logic Core (LC)**: Standard supervised fine-tuning (SFT) with a process reward signal on intermediate reasoning steps. This is the control condition — a well-trained reasoning model without GFP-inspired training.
- **Chaos Core (CC)**: Surprise-gated training seeded by Phase 2 traces. The training curriculum is shaped by prediction error from the Phase 2 LSTM: examples that were high-error (unexpected) in the GFP signal domain are up-weighted in the language model training. The surprise gate from Phase 2 is repurposed as a curriculum weighting signal.
- **Arbiter routing**: A lightweight MLP classifier is trained to route test-time queries to either Logic Core or Chaos Core based on predicted downstream task success. The Arbiter is trained on held-out validation tasks.

Evaluation: 6 BIG-Bench Hard (BBH) tasks, selected for coverage of causal reasoning, temporal sequences, and logical deduction. Reported metric: exact-match accuracy.

### Expected Observations (Highly Speculative)

These expectations are weakly motivated given the conceptual distance between Phase 2 (synthetic signals) and Phase 3 (language reasoning).

- If the surprise-gate mechanism transfers, Chaos Core may show improved performance on BBH tasks that require detecting early disambiguating signals in the prompt (causal reasoning, temporal sequences).
- Logic Core is expected to perform better on purely deductive tasks where the training distribution closely matches the fine-tuning data.
- The Arbiter may show mixed results — its value depends on whether Logic Core and Chaos Core are genuinely complementary rather than one being uniformly better.
- A negative result (LC ≥ CC across all tasks) would be informative: it would suggest that GFP-style training does not transfer to language, and would redirect Phase 4 back to the signal domain.

### Open Questions for Phase 3

1. How should Phase 2 traces be mapped to language training signal? The mapping from signal prediction error to language curriculum weighting is a design choice that may significantly affect results.
2. What is the appropriate baseline for the Arbiter? (A random router? An oracle router? An always-LC or always-CC router?)
3. Which BBH tasks are most likely to benefit from surprise-gate training? Speculative hypothesis: causal judgment, temporal sequences, and ruin names (which require detecting early contextual cues) are more likely to benefit than purely logical tasks (e.g., boolean expressions, formal fallacies).

---

## Phase 4 — TBD

**Status**: Not yet scoped  
**Decision point**: Phase 4 direction depends on Phase 3 outcomes and Phase 2 effect sizes.

See [PHASE4_OPTIONS.md](PHASE4_OPTIONS.md) for a detailed decision framework covering four candidate directions:
- 4A: Scale to Mistral-7B + LoRA
- 4B: Domain investigation (BBH task analysis)
- 4C: Interpretability (attention visualization, mechanistic analysis)
- 4D: Return to signal environment (multi-frequency, non-stationary, partial observability)

The decision criteria and expected conditions under which each option is appropriate are documented there.

---

*Log entries are written at phase completion. Pending phase entries will be updated with observed results when runs are complete.*
