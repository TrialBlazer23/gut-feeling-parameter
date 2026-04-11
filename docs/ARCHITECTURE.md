# Architecture Documentation — GFP Research

**Repository**: gut-feeling-parameter (Necessity Labs / TrialBlazer23)  
**Status**: Reflects Phase 1 (implemented) and Phase 2–3 (written, not yet run) components

---

## Table of Contents

1. [SignalEnvironment](#signalenvironment)
2. [PredictiveEncoder](#predictiveencoder)
3. [EpisodicMemory](#episodicmemory)
4. [ConfigD_Agent](#configd_agent)
5. [ConfigE_Agent](#confige_agent)
6. [Phase 3 Components](#phase-3-components)
7. [Hyperparameter Defaults and Justifications](#hyperparameter-defaults-and-justifications)

---

## SignalEnvironment

The `SignalEnvironment` is a custom OpenAI-Gym-compatible environment that generates synthetic time series with three distinct state types.

### State Types

| State Type | Description | Frequency | Duration |
|---|---|---|---|
| `NORMAL` | Low-amplitude, quasi-periodic signal; baseline state | ~75–80% of timesteps | Variable |
| `PRECURSOR` | Structured deviation preceding an event; the target of detection | ~5–10% of timesteps | 3–10 steps before event |
| `EVENT` | High-intensity burst; the "threat" being protected against | ~2–5% of timesteps | 1–5 steps |

### Signal Generation

The environment generates a 1D scalar signal at each timestep. The generative process differs by state type:

- **NORMAL**: Gaussian noise with low variance (σ ≈ 0.1) around a slow sinusoidal baseline. The baseline oscillation period is randomized per episode to prevent the agent from exploiting fixed timing.
- **PRECURSOR**: A structured ramp or oscillation pattern that is statistically distinguishable from normal states but not grossly different in amplitude. Designed so that a simple amplitude threshold will not reliably separate precursor from normal.
- **EVENT**: High-amplitude spike (amplitude ~3–5 × normal baseline). Always follows a precursor sequence of configurable length.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `episode_length` | 200 | Timesteps per episode |
| `precursor_length` | 5 | Number of precursor timesteps before each event |
| `event_prob` | 0.02 | Per-timestep probability of beginning an event sequence |
| `normal_noise_std` | 0.1 | Standard deviation of noise on normal states |
| `event_amplitude` | 4.0 | Signal amplitude during event state |
| `precursor_amplitude` | 1.5 | Peak amplitude during precursor sequence |
| `seed` | None | Random seed for reproducibility |

### Observation and Action Space

- **Observation**: A vector of dimension `obs_dim` (default: 1 — the current signal value). Phase 2 may extend to multi-dimensional observations.
- **Action space**: Discrete, 2 actions — `0` (do nothing) and `1` (flag as anomalous / take protective action).
- **Reward**: Configuration-dependent (see ConfigD_Agent and ConfigE_Agent sections).

### State Labels

State labels (NORMAL, PRECURSOR, EVENT) are available in the environment for evaluation purposes but are **not passed to the agent** in any configuration except Config A, which receives binary threat/normal labels (where threat = EVENT; PRECURSOR is withheld).

---

## PredictiveEncoder

The `PredictiveEncoder` is the core neural network module shared across all configurations. It combines an LSTM encoder, a predictive head for world modeling, and a policy head for action selection.

### Architecture

```
Input: observation [batch, seq, obs_dim]
         │
         ▼
┌─────────────────────┐
│    LSTM Encoder     │  layers=1–2, hidden_size=32–256
└─────────────────────┘
         │ hidden state h_t
         ├──────────────────────────────┐
         ▼                              ▼
┌──────────────────┐        ┌───────────────────────┐
│  Predictive Head │        │     Policy Head        │
│  (next-step      │        │  (action logits,       │
│   prediction)    │        │   entropy regularized) │
└──────────────────┘        └───────────────────────┘
         │                              │
    pred_error_t                   action_probs_t
```

### LSTM Encoder

A standard PyTorch `nn.LSTM` module.

| Hyperparameter | Phase 1 | Phase 2 | Justification |
|---|---|---|---|
| `input_size` | obs_dim (1) | obs_dim (1) | Matches environment observation dimension |
| `hidden_size` | 32–64 | 256 | Small in Phase 1 for speed; 256 in Phase 2 to give enough capacity for precursor representation learning |
| `num_layers` | 1 | 1–2 | Single layer in Phase 1; 2 layers optional in Phase 2 |
| `dropout` | 0 | 0.1 | Small dropout in Phase 2 to reduce overfitting across seeds |

### Predictive Head

A two-layer MLP that maps the LSTM hidden state to a scalar prediction of the next timestep's signal value.

```
h_t → Linear(hidden_size, hidden_size // 2) → ReLU → Linear(hidden_size // 2, obs_dim) → predicted_obs_{t+1}
```

- **Loss** (used internally for world model training, not agent reward): MSE(predicted_obs, actual_obs).
- **Prediction error** `pred_error_t = |predicted_obs_{t+1} − actual_obs_{t+1}|` (L1 error). This is the primary internal signal used by the intrinsic motivation system.
- The predictive head is trained jointly with the policy head via gradient flow through the LSTM.

### Policy Head

A two-layer MLP that maps the LSTM hidden state (optionally concatenated with memory retrieval vector in Config E) to action logits.

```
h_t → Linear(hidden_size, hidden_size // 2) → ReLU → Linear(hidden_size // 2, n_actions=2) → logits
```

- **Policy gradient**: REINFORCE with per-episode return normalization.
- **Entropy bonus**: Added to the policy loss to prevent premature collapse to deterministic actions. Coefficient: `entropy_coeff` (default 0.01).
- **Gradient clipping**: Global norm clipping at `max_grad_norm` (default 1.0) applied before each optimizer step.

---

## EpisodicMemory

The `EpisodicMemory` module is used in Config E only. It provides a simple episodic key-value store with k-NN retrieval via cosine similarity.

### Storage

At each timestep, the agent optionally writes a memory entry consisting of:
- **Key**: The LSTM hidden state `h_t` at the time of writing (dimension: `hidden_size`)
- **Value**: The prediction error `pred_error_t` at the time of writing (scalar)
- **Write condition**: Memory is written when `pred_error_t > mean + 1σ` (same threshold as surprise gate), to ensure the episodic buffer represents high-error states.

Memory entries are stored in a fixed-capacity ring buffer (default capacity: 1000 entries). Older entries are overwritten when the buffer is full.

### Retrieval

At each decision step, the agent retrieves the k nearest memory entries by cosine similarity between `h_t` and all stored keys.

```python
similarities = cosine_similarity(h_t, memory_keys)  # [memory_size]
top_k_indices = argsort(similarities, descending=True)[:k]
retrieved_values = memory_values[top_k_indices]       # [k]
```

The retrieved values (prediction errors of similar past states) are aggregated via a **learned gate**:

```python
gate_input = concat([h_t, mean(retrieved_values)])    # [hidden_size + 1]
gate = Linear(hidden_size + 1, 1) → Sigmoid           # scalar in [0, 1]
memory_signal = gate * mean(retrieved_values)         # scalar
```

The memory signal is concatenated with `h_t` and passed to the policy head:

```python
policy_input = concat([h_t, memory_signal])           # [hidden_size + 1]
```

### Rationale

The episodic memory hypothesis is that if the agent has previously encountered states similar to the current one (by LSTM representation) and those states were high-error, that is evidence that the current state is also a high-error (precursor-like) state. The learned gate allows the agent to suppress the memory signal if it is not useful.

### Parameters

| Parameter | Default | Justification |
|---|---|---|
| `memory_capacity` | 1000 | Large enough to store multiple episodes' worth of high-error states |
| `k` | 5 | Small k for stability; larger k risks averaging over irrelevant memories |
| `similarity_metric` | cosine | Scale-invariant; appropriate for normalized hidden states |

---

## ConfigD_Agent

`ConfigD_Agent` implements the pure curiosity configuration. This is the primary experimental agent.

### Surprise Gate Formula

The reward at timestep `t` is:

```
r_t = 0.5 * normalize(pred_error_t)   if action_t == 1 AND pred_error_t > μ_t + σ_t
    = 0                                otherwise
```

Where:
- `normalize(x) = (x − min_error) / (max_error − min_error + ε)` clips prediction error to [0, 1] using episode running stats
- `μ_t` is the running mean of prediction errors up to timestep `t`
- `σ_t` is the running standard deviation of prediction errors up to timestep `t`
- Both are computed via the Welford online algorithm (see below)
- `ε = 1e-8` to prevent division by zero

### Welford Online Algorithm

Running mean and variance are updated at each timestep using Welford's numerically stable algorithm:

```python
def welford_update(count, mean, M2, new_value):
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    variance = M2 / count if count > 1 else 0.0
    std = sqrt(variance + 1e-8)
    return count, mean, M2, std
```

This is reset at the start of each episode. It produces a per-episode adaptive threshold without any fixed hyperparameter for the baseline level of prediction error.

### Policy Training

REINFORCE with Monte Carlo returns:

```
G_t = Σ_{t'=t}^{T} γ^{t'−t} r_{t'}

L_policy = − Σ_t G_t * log π(a_t | s_t) − entropy_coeff * H(π)
```

Where `H(π) = − Σ_a π(a) log π(a)` is the policy entropy.

Returns are normalized per episode (subtract mean, divide by std + ε) before computing the policy gradient.

### Training Loop

For each episode:
1. Reset environment and Welford stats.
2. Roll out full episode, collecting `(h_t, a_t, pred_error_t, r_t)`.
3. Compute Monte Carlo returns.
4. Compute policy loss + entropy bonus.
5. Backpropagate through policy head and LSTM encoder.
6. Clip gradients by global norm.
7. Optimizer step (Adam).

The predictive head is trained concurrently with the policy head via the MSE world model loss, backpropagated at each timestep.

### Parameters

| Parameter | Default | Justification |
|---|---|---|
| `lr` | 1e-3 | Standard starting LR for Adam in RL |
| `gamma` | 0.99 | High discount; episodes are short (200 steps) |
| `entropy_coeff` | 0.01 | Small enough not to overwhelm the intrinsic signal |
| `max_grad_norm` | 1.0 | Standard REINFORCE stability measure |
| `reward_scale` | 0.5 | Keeps reward in a reasonable range relative to policy entropy |
| `welford_threshold_multiplier` | 1.0 | Mean + 1σ; see HYPOTHESIS.md for ablation discussion |

---

## ConfigE_Agent

`ConfigE_Agent` extends `ConfigD_Agent` with episodic memory. The policy receives both the LSTM hidden state and the memory retrieval signal.

### Extended Policy Input

```python
# At each step:
memory_signal = episodic_memory.retrieve(h_t)       # uses k-NN cosine similarity
policy_input = concat([h_t, memory_signal])          # [hidden_size + 1]
logits = policy_head(policy_input)
```

After each step, if `pred_error_t > μ_t + σ_t`:

```python
episodic_memory.write(key=h_t.detach(), value=pred_error_t.detach())
```

The `.detach()` ensures that memory writes do not create gradient loops through stored tensors.

### Reward Structure

Identical to Config D:

```
r_t = 0.5 * normalize(pred_error_t)   if action_t == 1 AND pred_error_t > μ_t + σ_t
    = 0                                otherwise
```

The memory signal influences the policy (what action to take) but not the reward gate (what triggers reward). This separation is intentional: the memory provides a prediction of future surprise, but the reward is still tied to the agent's live prediction error.

### Parameters

Inherits all Config D parameters, plus:

| Parameter | Default | Justification |
|---|---|---|
| `memory_capacity` | 1000 | See EpisodicMemory section |
| `k_nearest` | 5 | See EpisodicMemory section |
| `gate_lr` | 1e-3 | Gate is a small MLP, same LR as policy |

---

## Phase 3 Components

Phase 3 components are implemented in `training/v4_phase3_lm/phase3_chaos_core_lm.py`. They are written but not yet run.

### LogicCoreDataset

Wraps a subset of BIG-Bench Hard tasks formatted for standard SFT. Each example is a (prompt, chain-of-thought, final answer) triple. Process reward is applied at intermediate reasoning steps: each step that is factually consistent with the final correct answer receives a small positive reward signal.

### ChaosCoreDataset

Extends `LogicCoreDataset` with a curriculum weighting mechanism derived from Phase 2 GFP traces. Examples are weighted by the mean prediction error they receive from a frozen Config D agent that processes a text encoding of the problem (via a shared tokenizer). The intuition is that problems which are "surprising" to the GFP signal domain model may share structural properties (e.g., early disambiguating cues, non-monotonic context) that are relevant to reasoning difficulty.

**Design note**: The mapping from signal prediction error to language curriculum weight is a design choice that has not been empirically validated. It is motivated by the Phase 2 finding (if confirmed) that the GFP agent develops sensitivity to structured early-signal deviations. Whether this property transfers to language problem structure is an open question.

### Arbiter

A lightweight MLP (2 hidden layers, hidden_size=64) that receives a fixed-length embedding of the input query (from the GPT-2 encoder, frozen) and outputs a routing probability: `p(route_to_chaos_core | query)`.

Training: supervised on a held-out validation set, where labels are assigned by whichever core (Logic or Chaos) achieved correct final answers. Cross-entropy loss.

At test time, the Arbiter routes each query to either Logic Core or Chaos Core based on `p > 0.5`.

### GFP Routing (Phase 3 Inference)

```
Input query
     │
     ▼
GPT-2 encoder (frozen)
     │
     ▼
Arbiter MLP
     │
     ├── p > 0.5 → Chaos Core (GPT-2 + surprise-gated fine-tuning)
     └── p ≤ 0.5 → Logic Core (GPT-2 + SFT + process reward)
```

---

## Hyperparameter Defaults and Justifications

### Summary Table

| Hyperparameter | Value (Phase 1) | Value (Phase 2) | Justification |
|---|---|---|---|
| `hidden_size` | 32–64 | 256 | Phase 1: speed; Phase 2: capacity for representation learning |
| `num_layers` | 1 | 1–2 | Added depth in Phase 2 as optional ablation |
| `lr` | 1e-3 | 1e-3 | Standard starting point; not tuned in Phase 1 |
| `gamma` | 0.99 | 0.99 | Appropriate for 200-step episodes with delayed structure |
| `entropy_coeff` | 0.01 | 0.01 | Empirically standard; prevents premature collapse |
| `max_grad_norm` | 1.0 | 1.0 | Standard for REINFORCE stability |
| `reward_scale` | 0.5 | 0.5 | Keeps intrinsic reward below policy entropy scale |
| `welford_multiplier` | 1.0 | 1.0 (ablated) | Mean + 1σ; ablated in Phase 2 at 0.5, 1.5, 2.0 |
| `memory_capacity` | N/A | 1000 | Config E only |
| `k_nearest` | N/A | 5 | Config E only |
| `episodes` | 50 | 200 | Phase 2: long enough for LSTM convergence at hidden_size=256 |
| `seeds` | 1 | 5 | Phase 2: minimum for Welch t-test |

### Hyperparameter Sensitivity Notes

- **`welford_multiplier`**: This is the most consequential hyperparameter for Config D. A lower multiplier makes the gate fire more often (more false alarms possible, more gradient signal). A higher multiplier makes the gate fire rarely (only very strong prediction errors trigger reward). The Phase 1 default of 1.0 was not tuned; Phase 2 ablation is important.
- **`entropy_coeff`**: If too large, the policy collapses to uniform action sampling. If too small, the policy may collapse to always action == 0 (if action == 1 generates penalty in some episode configurations) or always action == 1 (reward hacking). The chosen value (0.01) is standard but warrants monitoring.
- **`gamma`**: At 0.99 with 200-step episodes, the effective horizon is ~100 steps. This is appropriate for credit assignment when precursor states are 3–10 steps before events.

---

*See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for per-phase run records and [HYPOTHESIS.md](HYPOTHESIS.md) for the formal scientific claims these components are designed to test.*
