# Related Work — GFP Research

**Repository**: gut-feeling-parameter (Necessity Labs / TrialBlazer23)  
**Purpose**: A literature review situating the GFP mechanism relative to prior work. For each paper, we describe what it found, what GFP borrows from it, and where GFP differs.

---

## Table of Contents

1. [Pathak et al. (2017) — Intrinsic Curiosity Module (ICM)](#pathak-et-al-2017--intrinsic-curiosity-module-icm)
2. [Burda et al. (2018) — Random Network Distillation (RND)](#burda-et-al-2018--random-network-distillation-rnd)
3. [Wayne et al. (2018) — MERLIN](#wayne-et-al-2018--merlin)
4. [Baker et al. (2019) — Emergent Tool Use](#baker-et-al-2019--emergent-tool-use)
5. [Jhin et al. (2023) — Precursor of Anomaly (PoA)](#jhin-et-al-2023--precursor-of-anomaly-poa)
6. [Friston (2009) — Free Energy Principle](#friston-2009--free-energy-principle)
7. [What GFP Does That None of These Do](#what-gfp-does-that-none-of-these-do)

---

## Pathak et al. (2017) — Intrinsic Curiosity Module (ICM)

**Full citation**:  
Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 70, 2778–2787.  
arXiv: [1705.05363](https://arxiv.org/abs/1705.05363)

### What It Found

ICM introduces a framework for intrinsic motivation in which the agent's reward is augmented by a prediction error signal derived from a forward model of the environment. The forward model predicts the next state embedding given the current state embedding and the action taken. The prediction error — the gap between the predicted and actual next-state embedding — serves as an intrinsic bonus: high error means the transition was surprising and therefore worth exploring.

Key results: ICM-augmented agents learn to explore efficiently in sparse-reward environments (VizDoom, Super Mario Bros.) and outperform agents with only extrinsic reward in settings with long stretches of zero reward. The feature encoding step (projecting states through an inverse dynamics model before using them in the forward model) was shown to be important for filtering out irrelevant sources of prediction error (e.g., flickering screen noise).

### How It Relates to GFP

**What GFP borrows**: The core idea — that prediction error from an internal world model can serve as a useful intrinsic reward — is directly adopted. GFP's predictive head and the use of `pred_error_t` as the raw intrinsic signal are downstream of ICM's contribution. GFP also follows ICM in training the forward model jointly with the policy rather than as a separate pre-training stage.

**Where GFP differs**:  
1. **Action-gating**: ICM applies the prediction-error bonus continuously as an additive reward, regardless of what action the agent takes. GFP's Config D conditions the intrinsic reward on a specific action (`action == 1`) and a threshold gate. The agent must choose to "act on" the surprise, not merely experience it. This creates a qualitatively different learning signal.  
2. **Threshold gating**: ICM uses prediction error directly as a scalar bonus; GFP normalizes against a running per-episode baseline (Welford). This makes the threshold adaptive rather than fixed.  
3. **Target environment**: ICM was demonstrated in video game navigation environments; GFP is applied to a 1D temporal signal detection task and (in Phase 3) language reasoning. The environments are structurally different, and the ICM results do not directly predict GFP's behavior.
4. **Evaluation focus**: ICM evaluates exploration efficiency (state coverage, reward achieved). GFP evaluates precursor detection recall/precision — a different downstream question.

---

## Burda et al. (2018) — Random Network Distillation (RND)

**Full citation**:  
Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by random network distillation. *International Conference on Learning Representations (ICLR 2019)*.  
arXiv: [1810.12894](https://arxiv.org/abs/1810.12894)

### What It Found

RND proposes a simpler and more scalable alternative to ICM for intrinsic motivation. A fixed, randomly initialized target network maps observations to feature vectors. A predictor network, trained to minimize the distance to the target network's outputs, produces prediction error as its intrinsic signal. Because the target network is random and fixed, prediction error is high for states that are novel (not yet seen frequently by the predictor) and low for familiar states. RND was shown to achieve state-of-the-art results on hard exploration benchmarks, including Montezuma's Revenge.

Key insight: the intrinsic signal is purely count-based (proportional to novelty of a state representation) and does not require learning a forward dynamics model. This sidesteps the problem of distractor features that ICM partially addresses through inverse dynamics encoding.

### How It Relates to GFP

**What GFP borrows**: GFP and RND share the general goal of producing a novelty-sensitive signal. Both use a form of "prediction error" as the operationalization of surprise, though the underlying mechanism differs (learned forward model vs. random network distillation).

**Where GFP differs**:  
1. **Dynamics modeling**: GFP uses a genuine forward model (predicts the next observation value), while RND's "prediction error" is entirely about state novelty with no dynamics. This means GFP's signal is sensitive to *temporal predictability*, not just state recurrence — a distinction that may matter for precursor detection, where the key signal is a temporal deviation from the learned dynamics.  
2. **Action-gating**: Same distinction as with ICM — RND applies the novelty bonus continuously, with no action-gating requirement.  
3. **Interpretability**: GFP's prediction error is interpretable in terms of the agent's temporal world model (how well it predicted the next observation). RND's error reflects the current state's distance from the distribution of previously seen states.  
4. **Scale**: RND was demonstrated at scale (Atari) with deep convolutional networks. GFP Phase 1 uses an LSTM on a synthetic 1D signal — the operating scale is vastly different and GFP's results cannot be compared to RND benchmarks.

---

## Wayne et al. (2018) — MERLIN

**Full citation**:  
Wayne, G., Hung, C.-C., Amos, D., Mirza, M., Ahuja, A., Rae, J., ... & Botvinick, M. (2018). Unsupervised predictive memory in a goal-directed agent. arXiv preprint.  
arXiv: [1803.10760](https://arxiv.org/abs/1803.10760)

### What It Found

MERLIN (Memory, RL, and Inference Network) integrates a world model, a working memory module, and an external episodic memory into a single agent architecture. The world model generates a compressed state representation and predicts future states; the episodic memory stores these representations and supports retrieval via learned attention. MERLIN was evaluated on complex 3D navigation and memory tasks (from the DeepMind Lab suite) that require both long-horizon memory and temporal reasoning.

Key results: MERLIN outperformed standard LSTM agents and simpler memory-augmented baselines on tasks requiring the agent to remember specific past events (e.g., "go back to the object you saw 30 steps ago"). The integration of world model and episodic memory was shown to be synergistic: the compressed world model representations were more useful as memory keys than raw observations.

### How It Relates to GFP

**What GFP borrows**: Config E's episodic k-NN memory is structurally inspired by the MERLIN memory module. Specifically: using the LSTM hidden state (a compressed representation trained via a world model) as the memory key, and retrieving by similarity to support future action selection, mirrors MERLIN's architecture at a smaller scale.

**Where GFP differs**:  
1. **Scope and scale**: MERLIN is a large, integrated system with multiple interacting modules, trained on 3D environments with rich visual inputs. Config E is a much smaller system with a single LSTM and a simple ring buffer. No claim is made that Config E approaches MERLIN in capability.  
2. **Memory write policy**: MERLIN writes to memory according to a learned write gate. Config E writes only at high-prediction-error moments (surprise-gated write). This selective write policy is specific to GFP's hypothesis that high-error states (i.e., precursor-like states) are the relevant ones to remember.  
3. **Goal structure**: MERLIN is a goal-directed agent. Config E has no goal (same as Config D). The episodic memory is used to improve the agent's ability to time its surprise-gated actions, not to assist in task completion.  
4. **Task**: MERLIN was evaluated on navigation and object recall tasks. Config E is evaluated on temporal signal precursor detection.

---

## Baker et al. (2019) — Emergent Tool Use

**Full citation**:  
Baker, B., Kanitscheider, I., Markov, T., Wu, Y., Powell, G., McGrew, B., & Mordatch, I. (2019). Emergent tool use from multi-agent autocurricula. *International Conference on Learning Representations (ICLR 2020)*.  
arXiv: [1909.07528](https://arxiv.org/abs/1909.07528)

### What It Found

Baker et al. demonstrated that agents trained in a multi-agent hide-and-seek environment with simple objectives (hiders rewarded for hiding, seekers for finding) spontaneously developed a sequence of increasingly sophisticated behaviors not explicitly programmed: box-carrying, ramp-blocking, ramp-surfing, and box-surfing. These emergent behaviors arose as each team adapted to the other's strategies, creating an autocurriculum of increasing complexity from simple reward signals.

Key insight: complex, structured behaviors can emerge from simple objectives when the environment is sufficiently rich and the agents interact in a competitive or cooperative dynamic. No explicit training signal for "use ramps" or "block objects" was required.

### How It Relates to GFP

**What GFP borrows conceptually**: The framing of GFP's main claim — that precursor sensitivity may *emerge* from curiosity-driven training without explicit precursor labels — is conceptually parallel to Baker et al.'s finding that complex behaviors emerge without explicit training for those behaviors. GFP invokes Baker et al. as motivation for the general idea that structured environments can produce structured behavior as a side effect of simple reward signals.

**Where GFP differs**:  
1. **Mechanism**: Baker et al.'s emergent complexity arises from multi-agent co-adaptation — the environment is effectively endogenous because the other agents change the difficulty. GFP operates with a single agent in a fixed environment; there is no multi-agent dynamic.  
2. **Nature of emergence**: Baker et al. observe emergent tool use — spatially and physically structured behaviors in a 3D environment. GFP claims emergent temporal sensitivity — the agent's action timing becomes correlated with a particular class of temporal patterns. These are qualitatively different types of emergence.  
3. **Scale and validation**: Baker et al.'s results are from large-scale GPU training with extensive controlled analysis. GFP's "emergence" claim rests on a single-run CPU pilot at a much smaller scale. The claim of emergence is correspondingly much weaker.

---

## Jhin et al. (2023) — Precursor of Anomaly (PoA)

**Full citation**:  
Jhin, S. Y., Lee, H., Jo, M., Kwon, J., Park, S., Park, S., & Hwang, S. J. (2023). Precursor of anomaly: a multivariate time series anomaly detection model based on a learnable graph. *IEEE Transactions on Knowledge and Data Engineering*, 36(4), 1413–1427.  
DOI: [10.1109/TKDE.2023.3249520](https://doi.org/10.1109/TKDE.2023.3249520)  
*(Note: please verify the exact publication details if citing formally — this citation is based on best available information.)*

### What It Found

The Precursor of Anomaly (PoA) model addresses anomaly detection in multivariate time series by explicitly modeling the pre-anomaly period as a distinct state class. PoA uses a graph neural network to capture inter-series dependencies and a temporal model to detect the structured deviations that precede full anomalies. Critically, PoA is a supervised model: it is trained with labels for both anomaly and precursor states.

Key results: PoA outperforms anomaly-detection baselines (LSTM-based, transformer-based) on several real-world time series anomaly benchmarks, specifically on early detection metrics that reward identifying the precursor before the anomaly manifests.

### How It Relates to GFP

**Direct relevance**: PoA is the most directly relevant prior work to GFP's core question. Both PoA and GFP are concerned with detecting precursor states in time series. PoA establishes that precursor detection is a well-defined problem and that explicit precursor supervision improves performance.

**What GFP borrows**: The problem framing — that anomaly precursors are a distinct, detectable state class worth targeting — is consistent with PoA's approach. GFP adopts a similar evaluation metric (precursor recall).

**Where GFP differs**:  
1. **Supervision**: PoA requires explicit precursor labels in training. GFP explicitly withholds precursor labels and asks whether an agent can develop precursor sensitivity without them. This is the key differentiator: GFP's question is whether unsupervised precursor sensitivity is achievable, not whether supervised precursor detection can be improved.  
2. **Mechanism**: PoA uses a graph-based temporal model with direct supervision. GFP uses surprise-gated RL with no precursor-relevant supervision signal.  
3. **Multivariate vs. univariate**: PoA operates on multivariate time series with learned inter-series dependencies. Phase 1 GFP uses a univariate signal. This is a significant scope difference.  
4. **Validation**: PoA has been validated on real-world benchmark datasets. GFP Phase 1 uses a synthetic environment. GFP does not claim to match or exceed PoA on any benchmark.

---

## Friston (2009) — Free Energy Principle

**Full citation**:  
Friston, K. (2009). The free-energy principle: a rough guide to the brain. *Trends in Cognitive Sciences*, 13(7), 293–301.  
DOI: [10.1016/j.tics.2009.04.005](https://doi.org/10.1016/j.tics.2009.04.005)

### What It Found

Friston's free energy principle proposes that biological systems minimize "free energy" — a quantity that upper-bounds the surprise (negative log probability) of sensory inputs given the organism's internal model. Minimizing free energy is equivalent to improving the internal model's predictions (reducing prediction error) while taking actions that steer sensory inputs toward expected states. This provides a unified account of perception, action, learning, and attention under a single variational framework.

Key insight: surprise minimization — the drive to be in states that are expected under the agent's generative model — may be a fundamental organizing principle of adaptive systems, not just an engineering trick.

### How It Relates to GFP

**Conceptual resonance**: GFP's predictive head can be loosely interpreted as a generative model of the signal. The prediction error that drives Config D's intrinsic reward is formally the surprise generated by the current state given the agent's model. In Friston's terms, the Config D agent is rewarded for detecting high-free-energy moments — states that its generative model finds surprising.

**Where GFP diverges from Friston**:  
1. **Direction of motivation**: Friston's framework motivates agents to *minimize* surprise (improve their models or seek expected states). Config D rewards agents for *acting on* surprise — the agent is not motivated to resolve uncertainty but to selectively respond to it. This is an inversion of the typical active inference framing.  
2. **Formalism**: Friston's framework is a rigorous variational Bayesian theory. GFP is an empirical RL experiment that uses prediction error as an engineered reward signal. GFP does not derive the reward function from the free energy principle; the connection is motivational and post-hoc.  
3. **Scope**: The free energy principle is a theory of biological cognition at the level of neural circuits, perception, and action. GFP is a small RL experiment. Invoking Friston as background motivation is appropriate; claiming GFP implements or tests the free energy principle would be an overstatement.

---

## What GFP Does That None of These Do

None of the five papers reviewed above — ICM, RND, MERLIN, Baker et al., or Friston — uses the specific mechanism that is the defining feature of Config D:

> **Action-gated surprise reward**: Intrinsic reward is only awarded when the agent simultaneously (a) takes a specific discrete action and (b) exceeds an adaptive prediction-error threshold derived from a running per-episode baseline.

The specifics that are jointly novel:

1. **ICM and RND** apply prediction error as a continuous additive reward. There is no action requirement — the agent receives the bonus regardless of what it does. The gate in Config D creates a selective pressure that is absent in both.

2. **MERLIN** uses a predictive world model and episodic memory (Config E extends in this direction) but is goal-directed and does not use action-gating on surprise as a reward mechanism.

3. **Baker et al.** demonstrate emergent complexity from simple objectives but via multi-agent dynamics in a 3D physical environment, not via a curiosity mechanism in a single-agent temporal prediction task.

4. **Friston** provides theoretical grounding for surprise minimization but does not propose or study the specific action-gating mechanism.

5. **PoA** explicitly supervises precursor detection; GFP's question is whether precursor sensitivity is achievable without that supervision.

**Important caveat**: The novelty claim here is about the specific combination — action-gating + adaptive threshold + REINFORCE policy gradient — and its application to temporal precursor detection without supervision. It is possible that papers exist in the broader RL exploration or anomaly detection literature that use similar mechanisms in different contexts. A more thorough literature search (including workshop papers, preprints, and anomaly detection surveys) is warranted before making strong novelty claims in any formal publication.

---

*For the experimental design motivated by this literature, see [HYPOTHESIS.md](HYPOTHESIS.md). For the architectural implementation, see [ARCHITECTURE.md](ARCHITECTURE.md).*
