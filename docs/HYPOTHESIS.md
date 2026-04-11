# GFP Hypothesis — Formal Statement

**Document version**: 1.0  
**Repository**: gut-feeling-parameter (Necessity Labs / TrialBlazer23)  
**Status**: Pilot data collected; hypothesis under test

---

## Table of Contents

1. [Formal Statement](#formal-statement)
2. [Mechanistic Prediction](#mechanistic-prediction)
3. [Falsification Criteria](#falsification-criteria)
4. [Confounds to Control For](#confounds-to-control-for)
5. [Relationship to Prior Work](#relationship-to-prior-work)

---

## Formal Statement

**Primary hypothesis (H1)**:  
An LSTM agent trained with a surprise-gated intrinsic reward — where reward is conditional on both taking a specific action and exceeding a running prediction-error threshold — will, without any explicit labels or objectives relating to precursor states, develop action selection biased toward those precursor states at a rate exceeding chance and exceeding a supervised baseline that receives no precursor labels.

**Secondary hypothesis (H2)**:  
The LSTM hidden states of a Config D agent will encode information about precursor state identity (detectable by a linear probe), even though the agent was never given precursor labels. This would suggest that the latent representation organizes around predictive structure, not around task-relevant labels.

**Tertiary hypothesis (H3)**:  
The surprise-gating mechanism — rather than a prediction-error reward applied continuously (as in ICM) — is causally responsible for precursor sensitivity. This implies that a version of Config D without action-gating but otherwise identical should show lower precursor detection.

**Note on scope**: H1 has pilot support (Config D outperforms Config A and C in Phase 1). H2 and H3 require Phase 2 experiments (linear probe; ablation of the action-gate). None of these hypotheses should be considered confirmed on the basis of Phase 1 data alone.

---

## Mechanistic Prediction

If the hypothesis holds, the following chain of events should be observable:

1. **World model calibration**: During training, the LSTM predictive head learns a baseline predictive model of the regular signal. Its prediction error is low on normal states and elevated on precursor states because precursor states break the statistical regularity the model has learned.

2. **Threshold emergence**: The Welford running statistics (online mean and standard deviation of prediction error) establish a per-episode adaptive baseline. Precursor-state prediction errors will tend to fall above mean + 1σ when the world model is well-calibrated; normal-state errors will tend to fall below it.

3. **Reward alignment**: Because reward is gated by action == 1 AND pred_error > mean + 1σ, the policy gradient signal reinforces taking action == 1 specifically at high-error moments. If step 2 holds, this disproportionately reinforces action == 1 at precursor states.

4. **Latent precursor encoding**: As the policy converges, the hidden state trajectories that reliably precede high-error moments will cluster in LSTM representation space, creating a detectable structure even without explicit precursor labels (H2).

5. **Action-gating matters**: If the gate is removed (continuous prediction error reward, no action requirement), the agent loses the selective pressure to *act* at high-error moments. The gradient signal becomes less discriminative, and precursor detection should drop (H3).

**Prediction for Phase 2**: Config D will show statistically significant precursor detection improvement over Config A across 5 seeds (Welch t-test p < 0.05, Cohen's d > 0.5). Config E (with episodic memory) may show further improvement if memory retrieval reinforces the precursor-state recognition signal.

**Prediction for Phase 3**: If precursor sensitivity is a general property of the surprise-gate mechanism, a language model trained with a surprise-gated curriculum (Chaos Core) should show improved performance on tasks that require attending to early contextual signals — specifically causal reasoning and temporal/sequential tasks in BIG-Bench Hard.

---

## Falsification Criteria

The following outcomes would count as evidence against the hypothesis, either by degree or outright:

| Falsification Scenario | What It Would Mean |
|---|---|
| Config D does not significantly outperform Config A in Phase 2 (p > 0.1, d < 0.2) | The pilot Config D result was likely noise or a single-run artifact; the hypothesis is not supported at this scale |
| Linear probe on Config D LSTM states shows precursor accuracy at or near chance | The agent is detecting precursors behaviorally by chance, not through learned representation; H2 is falsified |
| Config D (no gate) ablation shows equal or higher precursor detection than Config D | Action-gating is not causally responsible; the mechanism story is wrong even if H1 holds |
| Config D precursor detection is explained by raw signal amplitude alone (simple threshold) | A non-learning detector would suffice; the LSTM/RL machinery adds no value |
| Config D's "precursor detection" is primarily false alarms on high-amplitude normal states | The agent is detecting amplitude, not precursor structure; the metric is misleading |
| Phase 3 Chaos Core shows no improvement on causal/temporal BIG-Bench tasks vs Logic Core | The mechanism does not transfer to language; Phase 3 is negative |

**Important note on partial falsification**: A negative Phase 2 result does not falsify the general idea that surprise-gating can produce emergent sensitivity — it would mean the effect was not robust at the scale tested with this environment. A negative Phase 3 result would not rule out Phase 2's findings; it would mean the language transfer did not work as hypothesized. These are distinct claims and should be evaluated separately.

---

## Confounds to Control For

The following alternative explanations should be ruled out before attributing Phase 1 results to the hypothesized mechanism:

**1. Random action coincidence**  
A random policy that uniformly samples action == 1 with probability 0.5 would detect approximately 50% of any event class if detection is measured as recall only. Phase 2 should report both precision and recall, and compare against a random-action baseline.

**2. Signal amplitude confound**  
Precursor states may happen to be higher in raw amplitude than normal states in the synthetic environment. If so, any predictor that responds to amplitude would appear to detect precursors. Mitigation: check whether precursor states are statistically distinguishable from normal states in raw amplitude; if so, include an amplitude-threshold baseline.

**3. Temporal proximity confound**  
Precursor states immediately precede high-intensity events. If the agent's prediction error is elevated *after* the event (due to the event itself being surprising), and if there is temporal bleed in how detection windows are counted, the agent may appear to detect precursors while actually responding to the event aftermath. Mitigation: define precursor detection windows strictly as timesteps *before* event onset; confirm window boundaries in the evaluation code.

**4. Reward hacking**  
The agent may learn to always output action == 1 regardless of prediction error, thereby receiving reward whenever prediction error happens to exceed threshold by chance. Mitigation: monitor the base rate of action == 1 over training; if it converges to near 1.0, the policy has likely collapsed to this degenerate strategy.

**5. Hyperparameter sensitivity**  
The surprise-gate threshold (mean + 1σ) was not tuned and may happen to align well with the precursor signal statistics in this specific environment. Mitigation: Phase 2 should include sensitivity analysis over the threshold multiplier (0.5σ, 1σ, 1.5σ, 2σ) or run the Welford gate as a hyperparameter.

**6. Environment leakage**  
If the synthetic environment generator has any deterministic structure that correlates action-value timing with precursor onset (e.g., fixed episode structure), the agent might exploit this without genuinely detecting precursors. Mitigation: verify that episode seeds are truly random and that precursor onset timing is not periodic or correlated with episode boundaries.

---

## Relationship to Prior Work

The GFP hypothesis sits at the intersection of three research traditions:

**Intrinsic motivation / curiosity-driven RL**: Pathak et al. (2017) showed that prediction error can serve as a self-supervised reward signal enabling effective exploration. The GFP mechanism is downstream of this insight — the predictive head and intrinsic reward signal are directly inspired by ICM. The key departure is action-gating: ICM applies intrinsic reward continuously; GFP conditions it on a discrete action decision, creating a selective pressure that ICM lacks.

**Emergent complexity**: Baker et al. (2019) demonstrated that agents trained with simple objectives in multi-agent settings develop complex emergent behaviors not explicitly programmed. GFP's claim that precursor sensitivity may be emergent from curiosity is conceptually similar — it is not a claim about multi-agent dynamics, but it shares the idea that structured environments can produce structured representations as a side effect of simple reward signals.

**Episodic memory and world models**: Wayne et al. (2018) (MERLIN) showed that combining a world model with episodic memory improves performance on tasks requiring temporal reasoning. Config E extends Config D in this direction, testing whether retrieval of high-error memories improves the agent's ability to anticipate future high-error states (i.e., precursors).

**Free energy / surprise minimization**: Friston (2009) provides a theoretical framing in which minimizing surprise (prediction error) is a fundamental principle of adaptive systems. The GFP mechanism inverts this framing in an interesting way: instead of *minimizing* surprise, the agent is rewarded for *acting on* surprise — but the underlying mechanism (maintaining a generative model of the environment) is the same. Whether this inversion is theoretically principled or merely instrumental is an open question.

**What no prior work does**: None of the above — ICM, RND, MERLIN, Baker et al., Friston — uses an action-gated surprise reward where the agent receives intrinsic motivation only when it simultaneously acts and detects high prediction error. This specific conjunction is, to the best of our knowledge, novel. Whether it is *importantly* novel depends on whether it produces meaningfully different behavior, which is what Phase 2 is designed to test.

---

*See [RELATED_WORK.md](RELATED_WORK.md) for full paper citations and extended comparisons.*
