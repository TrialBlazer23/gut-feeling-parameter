# Gut Feeling Parameter (GFP) — Necessity Labs Research

**An investigation into emergent precursor sensitivity in curiosity-driven RL agents**

[![Status: Pilot Complete](https://img.shields.io/badge/status-Pilot%20Complete-yellow)](https://github.com/TrialBlazer23/gut-feeling-parameter)
[![GPU Experiments: Pending](https://img.shields.io/badge/GPU%20experiments-Pending-lightgrey)](https://github.com/TrialBlazer23/gut-feeling-parameter)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Abstract

This repository documents an ongoing investigation into whether a reinforcement learning agent, trained with no explicit objective and no labels identifying threat precursors, will spontaneously develop sensitivity to sub-threshold precursor signals through surprise-gated intrinsic motivation alone. In a CPU-scale pilot (Phase 1), a pure curiosity agent (Config D) detected precursor states at approximately 39% recall — outperforming a supervised baseline (Config A, 0%) and all reward-structured variants — suggesting that surprise-gated action selection may bias the agent toward anomalous precursor states it was never told to attend to. These results are preliminary: they were obtained in a synthetic signal environment on a single run per configuration, and the proposed mechanism has not yet been confirmed at scale or subjected to statistical testing. Phase 2 (multi-seed GPU replication) and Phase 3 (language model transfer) are written and scheduled; full conclusions await those results.

---

## Table of Contents

1. [Background](#background)
2. [Repository Structure](#repository-structure)
3. [Experimental Overview](#experimental-overview)
4. [Configuration Summary](#configuration-summary)
5. [Phase 1 Results (Pilot)](#phase-1-results-pilot)
6. [How to Run](#how-to-run)
7. [Known Limitations](#known-limitations)
8. [Related Work](#related-work)
9. [Citation](#citation)
10. [License](#license)

---

## Background

### The Gut Feeling Parameter Hypothesis

The Gut Feeling Parameter (GFP) hypothesis asks whether an agent that is never told what to protect, never given labels for precursors, and never given an external objective can nonetheless develop sensitivity to early-warning signals — purely as a side effect of being rewarded for noticing genuine surprises.

The setup is as follows. An environment generates a time series that is mostly regular but is occasionally preceded by a distinctive precursor pattern before a high-intensity event. A supervised agent told nothing about precursors fails to detect them because class imbalance makes the precursor class invisible during training. A curiosity-only agent, by contrast, receives reward when it takes a specific action (action == 1) at moments of high prediction error. The hypothesis is that precursor states — which break the regularity the agent has learned to predict — will produce elevated prediction error, and that this prediction error signal will act as a latent label that the agent stumbles upon without being told it exists.

### The Surprise-Gating Mechanism

The core novelty of this architecture relative to standard curiosity methods (ICM, RND) is **action-gating on surprise**. In ICM and RND, intrinsic reward flows continuously from the prediction error regardless of the agent's behavior. In Config D, reward is only awarded when the agent *acts* (action == 1) at a moment when prediction error exceeds a running threshold (mean + 1 standard deviation). This creates a selective pressure: the agent must learn both to predict the environment well enough that genuine surprises stand out, and to time its actions to coincide with those surprises. The mechanism resembles an internal "flag the anomaly" signal.

The running threshold is computed via Welford's online algorithm, making it parameter-free and adaptive to the statistics of each episode.

### Why This Is Interesting

If the finding holds at scale, it suggests that:

1. Precursor sensitivity may be an emergent byproduct of well-calibrated curiosity, not something that requires explicit supervision.
2. Action-gating on surprise may be a more structured inductive bias than continuous intrinsic reward, at least in environments with structured precursor dynamics.
3. The mechanism may generalize beyond synthetic signals — Phase 3 tests this on language model reasoning tasks.

None of these claims should be accepted on the basis of the pilot data alone. They are the motivating hypotheses that Phases 2 and 3 are designed to test.

---

## Repository Structure

| Path | Contents |
|---|---|
| `README.md` | This file — project overview, results summary, how to run |
| `docs/HYPOTHESIS.md` | Formal statement of the GFP hypothesis, falsification criteria, confounds |
| `docs/EXPERIMENT_LOG.md` | Running log of all phases with observed data and interpretation |
| `docs/ARCHITECTURE.md` | Technical documentation of all components and hyperparameters |
| `docs/RELATED_WORK.md` | Literature review — ICM, RND, MERLIN, PoA, free energy |
| `docs/PHASE4_OPTIONS.md` | Decision document for post-Phase-3 directions |
| `training/v2_phase1_cpu/` | Phase 1 CPU baseline scripts (Configs A–D) |
| `training/v3_phase2_gpu/` | Phase 2 GPU replication script (5 seeds, Config E added) |
| `training/v4_phase3_lm/` | Phase 3 language model application (GPT-2, Logic/Chaos Core) |
| `results/phase1/` | Phase 1 output logs and metrics |
| `results/phase2/` | Phase 2 outputs (pending) |
| `results/phase3/` | Phase 3 outputs (pending) |

---

## Experimental Overview

| Phase | Name | Status | Hardware | Description |
|---|---|---|---|---|
| Phase 1 | CPU Baseline Pilot | **Complete** | CPU (local) | Configs A–D, 1 seed each, 50 episodes, synthetic signal environment |
| Phase 2 | GPU Replication + Extension | **Pending** | A100 (Colab) | Configs A–E, 5 seeds, 200 episodes, Welch t-test, Cohen's d, t-SNE |
| Phase 3 | Language Model Transfer | **Pending** | A100 (Colab) | GPT-2, Logic Core vs Chaos Core, BIG-Bench Hard evaluation |
| Phase 4 | TBD (see PHASE4_OPTIONS.md) | **Not yet scoped** | TBD | Contingent on Phase 3 findings |

---

## Configuration Summary

| Config | Name | Reward Structure | Ward | Intrinsic Motivation | Phase |
|---|---|---|---|---|---|
| A | Supervised Baseline | Binary cross-entropy, threat/normal labels | None | None | 1, 2 |
| B | Protection Only | Ward health delta (REINFORCE) | Yes | None | 1, 2 |
| C | Protection + Intrinsic | Ward health + 0.5 × pred_error bonus | Yes | Prediction error bonus | 1, 2 |
| D | Intrinsic Only (Pure Curiosity) | 0.5 × norm_pred_error, gated by action==1 AND error > mean+1σ | None | Surprise gate (Welford) | 1, 2 |
| E | Intrinsic + Episodic Memory | Config D reward + k-NN episodic memory retrieval | None | Surprise gate + memory retrieval | 2 |

---

## Phase 1 Results (Pilot)

> **Caution**: These results are from a single run per configuration (no seed averaging), on a CPU, in a synthetic environment. They should be treated as pilot observations, not statistically verified findings.

| Config | Precursor Detection Rate | Ward Health (final) | Notes |
|---|---|---|---|
| A — Supervised Baseline | 0% | N/A | Class imbalance; precursor labels withheld; baseline fails entirely |
| B — Protection Only | ~36% | ~0 | False alarm penalty too punishing; ward collapses |
| C — Protection + Intrinsic | ~29% | Partial | False alarm penalty suppresses sensitivity relative to D |
| D — Intrinsic Only | **~39%** | 100 (no ward) | Cleanest result; outperforms all other configs |

**Key observation**: Config D, which has the least external structure (no ward, no objective, no threat/normal labels), produces the highest precursor detection rate in the pilot. This is consistent with the surprise-gating hypothesis but does not confirm it — single-run results could reflect random variation, hyperparameter sensitivity, or artifacts of the synthetic environment.

The mechanism as observed: surprise-gated action reward fires disproportionately during precursor states because those states generate above-threshold prediction error relative to the agent's learned baseline, effectively acting as a latent signal the agent exploits without being told it exists.

---

## How to Run

### Prerequisites

- Python 3.9+
- PyTorch 2.x
- For Phase 2/3: Google Colab with A100 GPU runtime (or equivalent)

### Phase 2 (GPU Replication)

```bash
# From the repository root
python training/v3_phase2_gpu/phase2_gpu_experiment.py
```

Expected runtime: ~2–4 hours on A100 (5 seeds × 200 episodes × 5 configs).

Outputs written to `results/phase2/`:
- Per-config, per-seed metrics (precursor detection rate, episode return, ward health)
- Welch t-test and Cohen's d comparing Config D vs others
- t-SNE embeddings of LSTM hidden states
- Linear probe accuracy on precursor identity from LSTM representations

### Phase 3 (Language Model)

```bash
# From the repository root
python training/v4_phase3_lm/phase3_chaos_core_lm.py
```

Expected runtime: ~4–8 hours on A100.

Outputs written to `results/phase3/`:
- BIG-Bench Hard accuracy by task for Logic Core, Chaos Core, and Arbiter-routed
- Training curves (SFT loss, process reward)
- Arbiter routing statistics

### Colab Setup (recommended for Phases 2–3)

1. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
2. Clone repo: `!git clone https://github.com/TrialBlazer23/gut-feeling-parameter`
3. Install deps: `!pip install torch transformers datasets scikit-learn`
4. Select A100 runtime (Runtime → Change runtime type → A100)
5. Run the relevant script from the cloned directory.

---

## Known Limitations

**Scale**: Phase 1 used a single seed per configuration. Reported percentages are from one run and may not reflect stable behavior. Phase 2 is explicitly designed to determine whether the pilot observations replicate.

**Environment**: The signal environment is synthetic. Precursor patterns are generated by a fixed stochastic process with known statistical structure. Generalization to real-world time series or language is unconfirmed.

**Detection metric**: "Precursor detection rate" is defined as the fraction of precursor-timestep windows in which the agent took action == 1. This is recall without precision weighting — a random agent that always takes action == 1 would also score 100% by this metric. The more informative metric (precision × recall / false alarm trade-off) will be computed in Phase 2.

**Mechanism vs. correlation**: The observation that Config D fires on precursor states is consistent with the surprise-gating story but does not confirm that the agent has learned any internal representation of precursors. The linear probe and t-SNE analysis in Phase 2 are intended to test whether precursor identity is actually encoded in the LSTM hidden states.

**Hyperparameter sensitivity**: Config D's performance may be sensitive to the threshold parameter (mean + 1σ), the entropy bonus coefficient, and the learning rate. Sensitivity analysis was not run in Phase 1.

**No comparison to simple baselines**: A random agent, a threshold-only detector operating on the raw signal, or a simple moving-average anomaly detector were not compared in Phase 1. Phase 2 should include at least one of these baselines.

---

## Related Work

| Paper | Relevance |
|---|---|
| Pathak et al. (2017). *Curiosity-driven Exploration by Self-Supervised Prediction*. ICML. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363) | ICM — intrinsic curiosity via prediction error. GFP borrows the prediction-error-as-reward idea; differs in action-gating. |
| Burda et al. (2018). *Large-Scale Study of Curiosity-Driven Learning*. ICLR. [arXiv:1808.04355](https://arxiv.org/abs/1808.04355) | RND — random network distillation as curiosity signal. Related family; no action-gating. |
| Wayne et al. (2018). *Unsupervised Predictive Memory in a Goal-Directed Agent (MERLIN)*. [arXiv:1803.10760](https://arxiv.org/abs/1803.10760) | Episodic memory + world model. Config E extends toward this direction. |
| Baker et al. (2019). *Emergent Tool Use from Multi-Agent Interaction*. ICLR. [arXiv:1909.07528](https://arxiv.org/abs/1909.07528) | Emergent complexity from simple objectives. Motivates GFP's claim that precursor sensitivity may be emergent. |
| Friston, K. (2009). *The free-energy principle: a rough guide to the brain*. Trends in Cognitive Sciences. [DOI:10.1016/j.tics.2009.04.005](https://doi.org/10.1016/j.tics.2009.04.005) | Theoretical grounding: surprise minimization as a general principle of adaptive systems. |

See [docs/RELATED_WORK.md](docs/RELATED_WORK.md) for detailed comparisons.

---

## Citation

If you use this work or build on it, please cite:

```bibtex
@misc{necessity2024gfp,
  author       = {Evan {TrialBlazer23} and {Necessity Labs}},
  title        = {{Gut Feeling Parameter (GFP): An Investigation into Emergent Precursor Sensitivity in Curiosity-Driven RL Agents}},
  year         = {2024},
  howpublished = {\url{https://github.com/TrialBlazer23/gut-feeling-parameter}},
  note         = {Pilot complete; GPU replication pending}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for full text.

---

*Necessity Labs — open research, measured claims.*
