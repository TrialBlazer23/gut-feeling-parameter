# Phase 4 Options — Decision Document

**Repository**: gut-feeling-parameter (Necessity Labs / TrialBlazer23)  
**Purpose**: Document the four candidate directions for Phase 4 work. Phase 4 has not been scoped; this document is a pre-analysis to be consulted after Phase 3 results are available.

**Decision principle**: Phase 4 direction should follow the evidence. The options are not equally motivated by all Phase 3 outcomes. This document specifies the conditions under which each option is appropriate, expected costs, and what questions each option is positioned to answer.

---

## Table of Contents

1. [Decision Framework](#decision-framework)
2. [Option 4A — Scale (Mistral-7B + LoRA)](#option-4a--scale-mistral-7b--lora)
3. [Option 4B — Domain Investigation](#option-4b--domain-investigation)
4. [Option 4C — Interpretability](#option-4c--interpretability)
5. [Option 4D — Return to Signals](#option-4d--return-to-signals)
6. [Summary Decision Matrix](#summary-decision-matrix)

---

## Decision Framework

Phase 3 can produce four qualitatively different outcomes, each of which maps to a different Phase 4 priority:

| Phase 3 Outcome | Interpretation | Recommended Phase 4 Direction |
|---|---|---|
| Chaos Core significantly outperforms Logic Core on ≥3 BBH tasks | GFP mechanism transfers to language; scaling may be informative | **4A (Scale) or 4B (Domain)** |
| Chaos Core outperforms Logic Core on specific task subcategories only | Mechanism has a narrower footprint than hoped; understand the domain | **4B (Domain Investigation)** |
| Chaos Core and Logic Core perform similarly overall | No clear transfer; mechanism may be too small-scale to matter at GPT-2 size | **4C (Interpretability) or 4D (Return to Signals)** |
| Logic Core outperforms Chaos Core uniformly | Surprise-gated curriculum hurts language training; mechanism does not transfer | **4D (Return to Signals)** |

Additionally: if Phase 2 statistical results are strong (large Cohen's d for Config D vs. others) but Phase 3 results are negative, that motivates 4D. If Phase 2 results are ambiguous, 4C (interpretability on Phase 2 representations) is warranted before further investment.

---

## Option 4A — Scale (Mistral-7B + LoRA)

### Description

Replace GPT-2 with Mistral-7B, fine-tuned via LoRA (Low-Rank Adaptation) for both Logic Core and Chaos Core. Replicate the Phase 3 experimental design at larger model scale with improved base capabilities.

### When to Choose

- Phase 3 shows Chaos Core improving over Logic Core on ≥3 BBH tasks with effect sizes that are practically meaningful (e.g., ≥5 percentage points).
- The hypothesis is that the Phase 3 result was partially masked by GPT-2's limited base capability, and that the GFP curriculum benefit would be more visible in a more capable base model.
- Do **not** choose 4A if Phase 3 Chaos Core results are uniformly negative — scaling a negative finding is expensive and unlikely to reverse the result.

### What to Modify

1. **Base model**: Replace GPT-2 (117M–345M parameters) with Mistral-7B-Instruct-v0.2.
2. **Fine-tuning**: Use LoRA (rank r=16, alpha=32, target modules: q_proj, v_proj) rather than full fine-tuning. This makes training feasible on a single A100 40GB.
3. **Phase 2 trace mapping**: The GFP curriculum signal derived from Phase 2 LSTM traces must be re-evaluated at larger scale. The mapping function (which maps signal prediction error to language curriculum weight) may need rescaling for the Mistral tokenizer and instruction format.
4. **Evaluation**: Expand BIG-Bench Hard evaluation from 6 to 20+ tasks. Include BIG-Bench Lite if compute allows.
5. **Arbiter**: Retrain on Mistral-scale embeddings; increase hidden size to 256.

### Expected Cost

| Resource | Estimate |
|---|---|
| A100 compute | ~20–40 GPU-hours |
| Colab Pro+ | Feasible with careful session management |
| Time | 2–4 weeks (including evaluation) |
| Risk | Moderate — larger model does not guarantee better transfer of the GFP mechanism |

### Expected Information Gain

If 4A succeeds (Chaos Core improves at Mistral scale), this is the strongest result in the project: it demonstrates that the surprise-gate mechanism generalizes across model scales in the language domain. If 4A fails (Chaos Core still does not outperform Logic Core), it suggests the GPT-2 Phase 3 result was not scale-limited, and the mechanism may genuinely not transfer.

---

## Option 4B — Domain Investigation

### Description

A targeted analysis of which BIG-Bench Hard subtasks and task properties are most (and least) associated with Chaos Core improvements. The goal is to understand the structure of the GFP mechanism's benefit in the language domain.

### When to Choose

- Phase 3 shows that Chaos Core outperforms Logic Core on some tasks but not others, with no clear overall winner.
- Phase 3 shows a positive overall trend but effect is small; task-level analysis may reveal a stronger signal in specific subtask families.
- Phase 2 shows strong precursor-sensitivity results for Config D; you want to understand which reasoning tasks have structural analogues to "precursor detection" in the language domain.

### Which BBH Subtasks to Focus On

The GFP mechanism is hypothesized to improve performance on tasks that share structural properties with temporal precursor detection: early disambiguating cues that predict the correct resolution, non-monotonic context in which the first part of the prompt changes meaning or salience at the end, and tasks requiring explicit temporal or causal ordering.

Suggested focus subtasks:

| BBH Task | Rationale |
|---|---|
| Causal Judgment | Requires detecting which event in a chain is the actual cause; analogous to precursor identification |
| Temporal Sequences | Explicit temporal ordering; matches Phase 2's signal-domain problem |
| Logical Deduction | Step-by-step constraint propagation; tests whether GFP helps with reasoning structure vs. just temporal sensitivity |
| Tracking Shuffled Objects | Maintaining state across multiple perturbation events; tests temporal memory |
| Word Sorting | Relatively simple ordering task; provides a contrast (GFP likely does not help here) |
| Ruin Names | Requires detecting the element of a phrase that makes it a "ruin"; analogous to finding the salient deviant signal |

### Hypothesis About Causal and Temporal Reasoning

The GFP mechanism rewards the agent for detecting structured deviations from its learned model of regularity. In the language domain, analogous "deviations" may be:

- Sentences that introduce a new causal agent or reverse expected causality
- Temporal markers that reorder expected narrative sequence
- Contextual cues early in a problem that predict the resolution but are easily missed

If this structural analogy holds, Chaos Core should show disproportionate improvement on Causal Judgment and Temporal Sequences relative to purely deductive tasks (Boolean expressions, formal fallacies). Testing this hypothesis requires task-level breakdown of Phase 3 results, which is why 4B is primarily an analysis option, not a new training option.

### Expected Cost

Relatively low: 4B primarily involves analyzing Phase 3 results more carefully and running targeted evaluations on additional BBH subtasks. Estimated additional compute: 5–10 GPU-hours for supplementary evaluation.

---

## Option 4C — Interpretability

### Description

A mechanistic analysis of what the GFP agent has learned, at both the Phase 2 (LSTM signal domain) and Phase 3 (language model) levels. The goal is to go beyond behavioral metrics and understand *how* the surprise-gate mechanism produces the observed behavior.

### When to Choose

- Phase 2 statistical results are ambiguous (small effect size, high variance across seeds) and it is unclear whether Config D has learned anything structural or is behaving superficially.
- Phase 3 results are negative, but Phase 2 results are positive — you want to understand what Phase 2 learned before deciding whether to return to the signal domain.
- Phase 3 results are positive — you want to understand *why* Chaos Core helps before scaling.
- You want to build the strongest possible case for the GFP mechanism before investing further compute.

### Phase 2 Interpretability Approach

| Analysis | Method | Expected Finding |
|---|---|---|
| Linear probe on LSTM hidden states | Train MLP on frozen Config D LSTM states; labels = precursor/normal/event | If above chance, confirms H2: agent implicitly encodes precursor identity |
| Activation patching | Intervene on specific LSTM gates at precursor vs. normal timesteps | Identifies which internal components are responsible for high prediction error on precursor states |
| Attention / gate analysis | Visualize LSTM forget/input gate activations as functions of timestep and state type | Characterizes what the LSTM "attends to" at precursor moments |
| Temporal ablation | Mask or zero the hidden state at various lags before precursor onset | Determines how many timesteps before a precursor the LSTM begins to deviate from normal-state dynamics |

### Phase 3 Interpretability Approach

| Analysis | Method | Expected Finding |
|---|---|---|
| Attention visualization | Extract attention weight matrices from Chaos Core vs. Logic Core on matched prompts | Tests whether Chaos Core distributes attention differently on early disambiguating tokens |
| Logit lens | Project intermediate layer representations onto the vocabulary at each layer | Characterizes where in the network Chaos Core's behavior diverges from Logic Core |
| Saliency mapping | Input × gradient attribution on BBH tasks where Chaos Core outperforms | Identifies which tokens receive higher attribution in Chaos Core responses |

### Tools

- TransformerLens (for GPT-2 circuit analysis)
- Captum (PyTorch attribution library)
- Custom activation visualization scripts (to be added to `training/` directory)

### Expected Cost

Phase 2 interpretability: ~2–5 GPU-hours (most analysis runs on saved checkpoints, not new training).  
Phase 3 interpretability: ~5–15 GPU-hours depending on breadth.

---

## Option 4D — Return to Signals

### Description

Return to the synthetic signal environment and test the GFP mechanism under more varied and challenging conditions. The goal is to understand the robustness and limits of the precursor-sensitivity finding before attempting further generalization.

### When to Choose

- Phase 3 results are negative (Language transfer does not work; GFP mechanism is domain-specific).
- Phase 2 results are positive but Phase 3 results are negative — this combination suggests the mechanism works in temporal signals but does not generalize to language, and the signal domain is the appropriate place for further investigation.
- Phase 2 results are positive but effect sizes are small — more challenging environments may produce stronger effect sizes and clearer mechanistic evidence.
- You want a more rigorous characterization of the mechanism before attempting language transfer again.

### Environment Variants to Test

**4D-1: Multi-frequency environment**  
Add multiple overlapping oscillation frequencies to the signal. Precursors are defined by deviations from a composite baseline rather than a single frequency. This tests whether the LSTM predictive head can learn a richer generative model and whether the surprise gate still fires selectively on precursors in a noisier environment.

*Hypothesis*: Config D's precursor detection rate will decrease with more frequencies (more complex baseline to learn) but remain above zero. The addition of episodic memory (Config E) may help more in this environment than in the simpler Phase 1 setting.

**4D-2: Non-stationary baseline**  
The signal's baseline statistics (mean, variance, frequency) drift over the course of an episode. Precursors are still defined relative to the local baseline. This tests the Welford adaptive threshold more directly: the running statistics must track a moving baseline.

*Hypothesis*: The Welford gate is adaptive by design and should handle non-stationarity better than a fixed threshold. But if the baseline drifts faster than the Welford estimator converges, the gate may miss precursors during high-drift periods.

**4D-3: Partial observability**  
The agent observes only a noisy, phase-shifted version of the underlying signal. The precursor pattern is present in the underlying signal but degraded in the observation. This tests whether the LSTM encoder can learn to denoise the observation sufficiently to produce useful prediction error.

*Hypothesis*: Performance will degrade relative to Phase 2 in proportion to the noise level. The linear probe analysis should show reduced (but potentially non-zero) precursor discriminability in LSTM representations.

**4D-4: Multiple precursor types**  
Introduce two or more distinct precursor patterns preceding different types of events. Tests whether the agent develops sensitivity to specific precursor types or a general "anomaly detector."

*Hypothesis*: This is the most interesting variant for the GFP claim. If Config D develops distinct clustering of multiple precursor types in LSTM representation space (visible via t-SNE), it suggests the agent is learning more than a generic anomaly detector. If all precursor types cluster together (indistinct from each other), the agent may be detecting "non-normal" rather than "this specific precursor."

### Expected Cost

| Variant | Compute | Time |
|---|---|---|
| 4D-1 (multi-frequency) | ~2–4 GPU-hours | 1 week |
| 4D-2 (non-stationary) | ~2–4 GPU-hours | 1 week |
| 4D-3 (partial observability) | ~2–4 GPU-hours | 1 week |
| 4D-4 (multiple precursor types) | ~4–8 GPU-hours | 2 weeks |

These are additive — running all four variants would require ~10–20 GPU-hours total.

---

## Summary Decision Matrix

| Condition | Recommended Option | Rationale |
|---|---|---|
| Phase 3: Chaos Core beats Logic Core on ≥3 tasks, strong effect | **4A (Scale)** | Evidence supports scaling; Mistral-7B is the natural next step |
| Phase 3: Chaos Core wins on specific task types only | **4B (Domain Investigation)** | Understand the mechanism's footprint before investing in scale |
| Phase 3: Mixed or null result; Phase 2 strong | **4C (Interpretability)** | Understand what Phase 2 learned before deciding on language vs. signal direction |
| Phase 3: Negative; Phase 2 strong | **4D (Return to Signals)** | Language transfer failed; signal domain is still interesting |
| Phase 2: Ambiguous or null | **4C (Interpretability) first, then revisit** | The mechanism may not be real at Phase 2 scale; interpretability will clarify |

**Note**: These recommendations are not exclusive. 4B analysis costs little and should probably be run alongside whichever of 4A, 4C, or 4D is selected as the primary direction. 4C's linear probe analysis for Phase 2 should be run regardless of Phase 3 outcome, as it tests H2 independently of the language transfer question.

---

*This document is a pre-analysis. It will be updated once Phase 3 results are available and a direction is selected.*
