# ============================================================
# Protection Drive — Phase 2 GPU Experiment
# Necessity Labs / Aletheia Framework
# ============================================================
# Run on: Colab Pro+ → A100 GPU
# Runtime: ~45 min (200 eps × 5 seeds × Configs D+E + analysis)
#
# Cell boundaries: # %%
# Run cells in order. Cell 1 must complete before any others.
# ============================================================

# %% ─── CELL 1 — INSTALLS & IMPORTS ─────────────────────────────────────────

# !pip install -q torch torchvision scikit-learn matplotlib seaborn scipy

import os
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Reproducibility helper
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./plots", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# %% ─── CELL 2 — ENVIRONMENT CONFIG ─────────────────────────────────────────

@dataclass
class EnvironmentConfig:
    n_steps: int = 1500
    window_size: int = 50
    base_mean: float = 0.0
    base_std: float = 1.0
    precursor_mean_shift: float = 0.08    # sub-threshold, hard to detect
    precursor_prob: float = 0.04          # ~4% of steps
    precursor_duration: int = 15
    threat_mean_shift: float = 0.35
    threat_prob: float = 0.015
    threat_duration: int = 8
    precursor_leads_threat: bool = True   # precursor predicts upcoming threat

@dataclass
class ExperimentConfig:
    n_episodes: int = 200
    n_seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    hidden_size: int = 256
    n_layers: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    # Config D parameters
    surprise_scale: float = 0.5           # reward = 0.5 * normalized_pred_error
    surprise_gate_std_mult: float = 1.0   # threshold = mean + 1.0*std
    # Config E parameters
    memory_capacity: int = 2000
    memory_k: int = 5                     # k-NN neighbors to retrieve
    memory_embed_dim: int = 64
    # Checkpoint / logging
    checkpoint_every: int = 50
    log_every: int = 10

ENV_CFG = EnvironmentConfig()
EXP_CFG = ExperimentConfig()

print("Environment config:")
print(json.dumps(asdict(ENV_CFG), indent=2))
print("\nExperiment config:")
print(json.dumps(asdict(EXP_CFG), indent=2))

# %% ─── CELL 3 — ENVIRONMENT ─────────────────────────────────────────────────

class SignalEnvironment:
    """
    1-D time-series environment with three state types:
      0 = normal  1 = precursor  2 = threat  3 = post-threat
    Agent observes a rolling window of the signal.
    No reward signal is provided — only the raw observation.
    """
    def __init__(self, cfg: EnvironmentConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)

    def _generate_episode(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (signal, state_labels) arrays of length n_steps."""
        cfg = self.cfg
        signal = self.rng.normal(cfg.base_mean, cfg.base_std, cfg.n_steps).astype(np.float32)
        labels = np.zeros(cfg.n_steps, dtype=np.int64)  # 0 = normal

        t = 0
        while t < cfg.n_steps:
            if self.rng.random() < cfg.precursor_prob and t + cfg.precursor_duration < cfg.n_steps:
                dur = cfg.precursor_duration
                signal[t:t+dur] += cfg.precursor_mean_shift
                labels[t:t+dur] = 1  # precursor
                t += dur
                # With probability 0.7, a threat follows within 5–20 steps
                if cfg.precursor_leads_threat and self.rng.random() < 0.70:
                    gap = self.rng.randint(5, 20)
                    start = t + gap
                    end = min(start + cfg.threat_duration, cfg.n_steps)
                    if start < cfg.n_steps:
                        signal[start:end] += cfg.threat_mean_shift
                        labels[start:end] = 2  # threat
                        if end < cfg.n_steps:
                            post_end = min(end + 5, cfg.n_steps)
                            labels[end:post_end] = 3  # post-threat
                        t = end
            else:
                t += 1

        return signal, labels

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._signal, self._labels = self._generate_episode()
        self._t = self.cfg.window_size
        obs = self._signal[:self.cfg.window_size]
        return obs, self._labels

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Return (next_obs, labels, done). No reward — agent provides its own."""
        cfg = self.cfg
        t = self._t
        obs = self._signal[t - cfg.window_size: t]
        true_label = self._labels[t - 1]
        done = (t >= cfg.n_steps)
        if not done:
            self._t += 1
        return obs, true_label, done

    def get_full_labels(self) -> np.ndarray:
        return self._labels.copy()

# %% ─── CELL 4 — AGENT ARCHITECTURES ────────────────────────────────────────

class PredictiveEncoder(nn.Module):
    """Shared LSTM encoder + prediction head used by both D and E."""
    def __init__(self, window_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (batch, window_size) raw signal window
        Returns: (hidden, pred_next, policy_logits)
        hidden: final LSTM hidden state (batch, hidden_size)
        """
        x = obs.unsqueeze(-1)  # (batch, window, 1)
        out, (h, c) = self.lstm(x)
        hidden = h[-1]                          # last-layer hidden (batch, hidden)
        pred_next = self.pred_head(hidden)      # predicted next value
        logits = self.policy_head(hidden)
        return hidden, pred_next, logits


class EpisodicMemory(nn.Module):
    """
    k-NN episodic memory indexed by prediction error magnitude.
    Stores (embed, true_label) pairs; retrieves top-k by cosine similarity.
    """
    def __init__(self, capacity: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.proj = nn.Linear(hidden_size, embed_dim)
        self.register_buffer('keys', torch.zeros(capacity, embed_dim))
        self.register_buffer('vals', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('filled', torch.tensor(0, dtype=torch.long))
        # Learned gate: how much to add memory context to policy
        self.gate = nn.Parameter(torch.tensor(0.1))

    def write(self, hidden: torch.Tensor, label: int):
        """Store a hidden state + its true label."""
        key = F.normalize(self.proj(hidden.detach()), dim=-1)  # (1, embed_dim)
        idx = int(self.ptr.item())

        self.keys = self.keys.clone()
        self.keys[idx] = key.squeeze(0).detach()
        self.vals[idx] = label
        self.ptr = (self.ptr + 1) % self.capacity
        self.filled = min(self.filled + 1, torch.tensor(self.capacity))

    def retrieve(self, hidden: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (retrieved_context, label_distribution) by cosine similarity.
        retrieved_context: (embed_dim,) weighted average of top-k keys
        label_distribution: (4,) soft label counts of top-k entries
        """
        n = int(self.filled.item())
        if n == 0:
            ctx = torch.zeros(self.embed_dim, device=hidden.device)
            dist = torch.zeros(4, device=hidden.device)
            return ctx, dist
        query = F.normalize(self.proj(hidden), dim=-1).squeeze(0)  # (embed_dim,)
        sims = torch.matmul(self.keys[:n], query)  # (n,)
        k_eff = min(k, n)
        top_idx = torch.topk(sims, k_eff).indices   # (k_eff,)
        top_keys = self.keys[top_idx]                # (k_eff, embed_dim)
        top_vals = self.vals[top_idx]                # (k_eff,)
        weights = F.softmax(sims[top_idx], dim=0)   # (k_eff,)
        ctx = (weights.unsqueeze(-1) * top_keys).sum(0)  # (embed_dim,)
        dist = torch.zeros(4, device=hidden.device)
        for i, v in enumerate(top_vals):
            dist[int(v.item())] += weights[i].item()
        return ctx, dist


class ConfigD_Agent(nn.Module):
    """
    Pure surprise-gated curiosity agent.
    Reward = 0.5 * normalized_pred_error IF pred_error > mean+1*std AND action==1
    No ward. No external objective.
    """
    def __init__(self, window_size: int, cfg: ExperimentConfig):
        super().__init__()
        self.encoder = PredictiveEncoder(window_size, cfg.hidden_size, cfg.n_layers)
        self.window_size = window_size
        self.cfg = cfg
        # Running stats for surprise normalization
        self.register_buffer('_err_mean', torch.tensor(0.0))
        self.register_buffer('_err_var', torch.tensor(1.0))
        self.register_buffer('_err_count', torch.tensor(0.0))

    def _update_running_stats(self, err: float):
        """Welford online algorithm."""
        self._err_count += 1
        n = self._err_count.item()
        delta = err - self._err_mean.item()
        self._err_mean += delta / n
        delta2 = err - self._err_mean.item()
        self._err_var += (delta * delta2 - self._err_var) / n

    def forward(self, obs: torch.Tensor):
        hidden, pred_next, logits = self.encoder(obs)
        return hidden, pred_next, logits

    def compute_intrinsic_reward(self, pred_next: torch.Tensor, actual_next: float, action: int) -> float:
        """
        reward = 0.5 * normalized_pred_error
        gated by: pred_error > running_mean + 1.0*std AND action == 1
        """
        pred_err = abs(pred_next.item() - actual_next)
        self._update_running_stats(pred_err)
        mean = self._err_mean.item()
        std = math.sqrt(max(self._err_var.item(), 1e-8))
        normalized = (pred_err - mean) / (std + 1e-8)
        threshold_exceeded = pred_err > (mean + self.cfg.surprise_gate_std_mult * std)
        if threshold_exceeded and action == 1:
            return self.cfg.surprise_scale * max(normalized, 0.0), pred_err, True
        return 0.0, pred_err, False


class ConfigE_Agent(nn.Module):
    """
    Config D + episodic k-NN memory retrieval.
    Memory context is gated and concatenated to policy input.
    """
    def __init__(self, window_size: int, cfg: ExperimentConfig):
        super().__init__()
        self.encoder = PredictiveEncoder(window_size, cfg.hidden_size, cfg.n_layers)
        self.memory = EpisodicMemory(cfg.memory_capacity, cfg.memory_embed_dim, cfg.hidden_size)
        self.window_size = window_size
        self.cfg = cfg
        # Extended policy head that takes hidden + memory context + label dist
        ext_in = cfg.hidden_size + cfg.memory_embed_dim + 4
        self.policy_ext = nn.Sequential(
            nn.Linear(ext_in, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 2)
        )
        self.register_buffer('_err_mean', torch.tensor(0.0))
        self.register_buffer('_err_var', torch.tensor(1.0))
        self.register_buffer('_err_count', torch.tensor(0.0))

    def _update_running_stats(self, err: float):
        self._err_count += 1
        n = self._err_count.item()
        delta = err - self._err_mean.item()
        self._err_mean += delta / n
        delta2 = err - self._err_mean.item()
        self._err_var += (delta * delta2 - self._err_var) / n

    def forward(self, obs: torch.Tensor, write_label: Optional[int] = None):
        hidden, pred_next, _ = self.encoder(obs)
        ctx, label_dist = self.memory.retrieve(hidden, self.cfg.memory_k)
        # Write to memory if label provided (after each step)
        if write_label is not None:
            self.memory.write(hidden, write_label)
        gate = torch.sigmoid(self.memory.gate)
        gated_ctx = gate * ctx.to(hidden.device)
        ext = torch.cat([hidden.squeeze(0), gated_ctx, label_dist.to(hidden.device)], dim=-1)
        logits = self.policy_ext(ext.unsqueeze(0))
        return hidden, pred_next, logits

    def compute_intrinsic_reward(self, pred_next: torch.Tensor, actual_next: float, action: int) -> Tuple[float, float, bool]:
        pred_err = abs(pred_next.item() - actual_next)
        self._update_running_stats(pred_err)
        mean = self._err_mean.item()
        std = math.sqrt(max(self._err_var.item(), 1e-8))
        normalized = (pred_err - mean) / (std + 1e-8)
        threshold_exceeded = pred_err > (mean + self.cfg.surprise_gate_std_mult * std)
        if threshold_exceeded and action == 1:
            return self.cfg.surprise_scale * max(normalized, 0.0), pred_err, True
        return 0.0, pred_err, False

# %% ─── CELL 5 — TRAINING LOOP ───────────────────────────────────────────────

def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def run_episode(agent, env: SignalEnvironment, optimizer: optim.Optimizer,
                cfg: ExperimentConfig, config_name: str, train: bool = True) -> Dict:
    """
    Run one episode. Returns metrics dict.
    Handles both ConfigD_Agent and ConfigE_Agent polymorphically.
    """
    obs, labels = env.reset()
    is_config_e = isinstance(agent, ConfigE_Agent)
    log_probs, rewards, pred_errors = [], [], []
    surprise_events = []  # (step_idx, pred_error, true_label, action)
    action_counts = {0: 0, 1: 0}
    precursor_detected = 0
    precursor_total = 0
    threat_detected = 0
    threat_total = 0
    all_hidden_states = []
    all_true_labels = []
    step = 0

    while True:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        if is_config_e:
            hidden, pred_next, logits = agent(obs_t, write_label=None)
        else:
            hidden, pred_next, logits = agent(obs_t)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_val = action.item()
        action_counts[action_val] += 1

        # Store hidden state for linear probe
        all_hidden_states.append(hidden.detach().cpu().squeeze(0).numpy())

        next_obs, true_label, done = env.step(action_val)
        all_true_labels.append(int(true_label))

        # Track per-state-type detection
        if true_label == 1:
            precursor_total += 1
            if action_val == 1:
                precursor_detected += 1
        elif true_label == 2:
            threat_total += 1
            if action_val == 1:
                threat_detected += 1

        # Intrinsic reward
        actual_next = float(next_obs[-1]) if len(next_obs) > 0 else 0.0
        reward, pred_err, is_surprise = agent.compute_intrinsic_reward(pred_next, actual_next, action_val)

        if is_surprise:
            surprise_events.append({
                "step": step,
                "pred_error": float(pred_err),
                "true_label": int(true_label),
                "action": action_val
            })

        # Write to Config E memory after step
        if is_config_e:
            with torch.no_grad():
                agent.memory.write(hidden, int(true_label))

        log_probs.append(log_prob)
        rewards.append(reward)
        pred_errors.append(pred_err)

        obs = next_obs
        step += 1

        if done:
            break

    # REINFORCE update
    if train and sum(abs(r) for r in rewards) > 1e-8:
        returns = compute_returns(rewards, cfg.gamma)
        returns_t = torch.FloatTensor(returns).to(DEVICE).detach()
        # Normalize returns
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        policy_loss = -sum(lp * r for lp, r in zip(log_probs, returns_t))
        # Entropy bonus
        entropy = -sum(lp * torch.exp(lp) for lp in log_probs) / len(log_probs)
        loss = policy_loss - cfg.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

    mean_err = float(np.mean(pred_errors)) if pred_errors else 0.0
    precursor_det_rate = precursor_detected / max(precursor_total, 1)
    threat_det_rate = threat_detected / max(threat_total, 1)
    surprise_rate = len(surprise_events) / max(step, 1)
    nonzero_reward_rate = sum(1 for r in rewards if r > 0) / max(step, 1)

    return {
        "config": config_name,
        "precursor_det_rate": precursor_det_rate,
        "threat_det_rate": threat_det_rate,
        "surprise_rate": surprise_rate,
        "precursor_base_rate": precursor_total / max(step, 1),
        "mean_pred_error": mean_err,
        "nonzero_reward_rate": nonzero_reward_rate,
        "action_rate": action_counts[1] / max(step, 1),
        "n_surprise_events": len(surprise_events),
        "surprise_events": surprise_events,
        "hidden_states": all_hidden_states,
        "true_labels": all_true_labels,
    }


def train_config(config_name: str, seed: int, cfg: ExperimentConfig,
                 env_cfg: EnvironmentConfig) -> List[Dict]:
    set_seed(seed)
    env = SignalEnvironment(env_cfg, seed=seed)

    if config_name == "D":
        agent = ConfigD_Agent(env_cfg.window_size, cfg).to(DEVICE)
    elif config_name == "E":
        agent = ConfigE_Agent(env_cfg.window_size, cfg).to(DEVICE)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)
    episode_metrics = []

    for ep in range(cfg.n_episodes):
        metrics = run_episode(agent, env, optimizer, cfg, config_name, train=True)
        metrics["episode"] = ep
        metrics["seed"] = seed
        # Keep only aggregate data in the per-episode record (not raw states)
        episode_record = {k: v for k, v in metrics.items()
                          if k not in ("hidden_states", "true_labels", "surprise_events")}
        episode_record["surprise_events"] = metrics["surprise_events"]
        episode_metrics.append(episode_record)

        if (ep + 1) % cfg.log_every == 0:
            print(f"  Config {config_name} seed {seed} ep {ep+1:3d}/{cfg.n_episodes} | "
                  f"prec_det={metrics['precursor_det_rate']:.3f} "
                  f"surp_rate={metrics['surprise_rate']:.4f} "
                  f"mean_err={metrics['mean_pred_error']:.4f}")

        # Save checkpoint
        if (ep + 1) % cfg.checkpoint_every == 0 or ep == cfg.n_episodes - 1:
            ckpt_path = f"./checkpoints/config{config_name}_seed{seed}_ep{ep+1}.pt"
            torch.save({
                "episode": ep + 1,
                "seed": seed,
                "config": config_name,
                "model_state": agent.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": episode_record,
            }, ckpt_path)

    # Final eval pass — collect hidden states for probe / t-SNE
    agent.eval()
    with torch.no_grad():
        final_metrics = run_episode(agent, env, optimizer, cfg, config_name, train=False)

    return episode_metrics, agent, final_metrics


print("Training functions defined.")

# %% ─── CELL 6 — MAIN TRAINING RUN ───────────────────────────────────────────

print("=" * 60)
print("PHASE 2 TRAINING — Configs D and E")
print(f"Seeds: {EXP_CFG.n_seeds}  |  Episodes: {EXP_CFG.n_episodes}  |  Hidden: {EXP_CFG.hidden_size}")
print("=" * 60)

all_results = {"D": {}, "E": {}}
final_agents = {"D": {}, "E": {}}
final_metrics_all = {"D": {}, "E": {}}

for config_name in ["D", "E"]:
    print(f"\n{'─'*40}")
    print(f"Training Config {config_name}")
    print(f"{'─'*40}")
    for seed in EXP_CFG.n_seeds:
        print(f"\n  Seed {seed}:")
        t0 = time.time()
        ep_metrics, agent, fin_metrics = train_config(config_name, seed, EXP_CFG, ENV_CFG)
        elapsed = time.time() - t0
        all_results[config_name][seed] = ep_metrics
        final_agents[config_name][seed] = agent
        final_metrics_all[config_name][seed] = fin_metrics
        print(f"  Seed {seed} complete in {elapsed:.1f}s | "
              f"final prec_det={ep_metrics[-1]['precursor_det_rate']:.3f}")

print("\nAll training complete.")

# %% ─── CELL 7 — STATISTICAL ANALYSIS (Welch t-test + Cohen's d) ─────────────

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = math.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / (pooled_std + 1e-10)


def get_final_n_ep_mean(results: Dict, config: str, seed: int, metric: str, n: int = 20) -> float:
    """Mean of last n episodes for a given metric."""
    eps = results[config][seed]
    vals = [e[metric] for e in eps[-n:]]
    return float(np.mean(vals))


metrics_to_test = ["precursor_det_rate", "threat_det_rate", "surprise_rate", "nonzero_reward_rate"]
stat_results = {}

for metric in metrics_to_test:
    d_vals = np.array([get_final_n_ep_mean(all_results, "D", s, metric) for s in EXP_CFG.n_seeds])
    e_vals = np.array([get_final_n_ep_mean(all_results, "E", s, metric) for s in EXP_CFG.n_seeds])

    t_stat, p_val = stats.ttest_ind(d_vals, e_vals, equal_var=False)
    d_effect = cohens_d(e_vals, d_vals)  # positive = E > D

    stat_results[metric] = {
        "D_mean": float(d_vals.mean()), "D_std": float(d_vals.std()),
        "E_mean": float(e_vals.mean()), "E_std": float(e_vals.std()),
        "t_stat": float(t_stat), "p_val": float(p_val),
        "cohens_d": float(d_effect),
        "significant": bool(p_val < 0.05)
    }

    sig = "*** p<0.05 ***" if p_val < 0.05 else "ns"
    print(f"\n{metric}:")
    print(f"  D: {d_vals.mean():.4f} ± {d_vals.std():.4f}")
    print(f"  E: {e_vals.mean():.4f} ± {e_vals.std():.4f}")
    print(f"  Welch t={t_stat:.3f}, p={p_val:.4f}  Cohen's d={d_effect:.3f}  {sig}")

# Save stats
with open("./results/statistical_analysis.json", "w") as f:
    json.dump(stat_results, f, indent=2)

print("\nStats saved to ./results/statistical_analysis.json")

# Primary success check
prec_stat = stat_results["precursor_det_rate"]
if prec_stat["significant"] and prec_stat["E_mean"] > prec_stat["D_mean"]:
    print("\n[SUCCESS] Config E significantly outperforms Config D on precursor detection (p < 0.05)")
elif prec_stat["p_val"] > 0.05:
    print(f"\n[NOTE] Config E does not significantly outperform D on precursor detection (p={prec_stat['p_val']:.4f})")
    print("  This is expected if memory needs longer warm-up. Consider N_EPS=300 or memory pre-population.")

# %% ─── CELL 8 — SURPRISE TRACE ANALYSIS ────────────────────────────────────

print("\n" + "=" * 60)
print("SURPRISE TRACE ANALYSIS")
print("=" * 60)

surprise_traces_output = {}

for config_name in ["D", "E"]:
    config_traces = {}
    for seed in EXP_CFG.n_seeds:
        episodes = all_results[config_name][seed]
        all_events = []
        for ep_data in episodes:
            for ev in ep_data.get("surprise_events", []):
                ev["episode"] = ep_data["episode"]
                all_events.append(ev)
        config_traces[str(seed)] = all_events
    surprise_traces_output[config_name] = config_traces

# Compute % of surprise events that are precursor states (label == 1)
for config_name in ["D", "E"]:
    total_events = 0
    precursor_events = 0
    threat_events = 0
    for seed in EXP_CFG.n_seeds:
        for ev in surprise_traces_output[config_name][str(seed)]:
            total_events += 1
            if ev["true_label"] == 1:
                precursor_events += 1
            elif ev["true_label"] == 2:
                threat_events += 1

    pct_precursor = precursor_events / max(total_events, 1) * 100
    pct_threat = threat_events / max(total_events, 1) * 100
    print(f"\nConfig {config_name}:")
    print(f"  Total surprise events: {total_events}")
    print(f"  % precursor state:     {pct_precursor:.1f}%  (target >35%)")
    print(f"  % threat state:        {pct_threat:.1f}%")
    print(f"  % normal/post:         {100 - pct_precursor - pct_threat:.1f}%")

    if pct_precursor < 20:
        print(f"  [WARNING] {config_name} precursor surprise rate <20% — agent not preferentially"
              " surprised by precursors. Investigate ENV_CFG.precursor_mean_shift or extend episodes.")

# Save traces
with open("./surprise_traces_phase2.json", "w") as f:
    json.dump(surprise_traces_output, f, indent=2)

print("\nSurprise traces saved to ./surprise_traces_phase2.json")
print(">>> DOWNLOAD THIS FILE before ending your Colab session <<<")

# %% ─── CELL 9 — LINEAR PROBE (Does D encode state structure?) ───────────────

print("\n" + "=" * 60)
print("LINEAR PROBE — Hidden State → State Type Classification")
print("=" * 60)

probe_results = {}

for config_name in ["D", "E"]:
    config_accs = []
    for seed in EXP_CFG.n_seeds:
        agent = final_agents[config_name][seed]
        fin = final_metrics_all[config_name][seed]
        hidden_states = np.array(fin["hidden_states"])   # (T, hidden_size)
        true_labels = np.array(fin["true_labels"])        # (T,)

        # Downsample if too large for sklearn (keep up to 5000 samples)
        if len(hidden_states) > 5000:
            idx = np.random.choice(len(hidden_states), 5000, replace=False)
            hidden_states = hidden_states[idx]
            true_labels = true_labels[idx]

        scaler = StandardScaler()
        X = scaler.fit_transform(hidden_states)
        y = true_labels

        # Stratified split
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0,
                                  solver='lbfgs', class_weight='balanced')
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        config_accs.append(acc)
        print(f"  Config {config_name} seed {seed}: probe accuracy = {acc:.3f}")

    mean_acc = float(np.mean(config_accs))
    probe_results[config_name] = {
        "per_seed_acc": config_accs,
        "mean_acc": mean_acc,
        "chance_level": 0.25
    }
    print(f"  Config {config_name} MEAN probe accuracy: {mean_acc:.3f}  (chance=0.25, target>0.45)")
    if mean_acc > 0.60:
        print(f"  [SUCCESS] Config {config_name} strongly encodes state structure (>0.60)")
    elif mean_acc > 0.45:
        print(f"  [SUCCESS] Config {config_name} moderately encodes state structure (>0.45)")
    elif mean_acc <= 0.25:
        print(f"  [FAILURE] Config {config_name} at chance. Extend to 200+ ep before probing.")

with open("./results/probe_results.json", "w") as f:
    json.dump(probe_results, f, indent=2)

# %% ─── CELL 10 — t-SNE VISUALIZATION ────────────────────────────────────────

print("\n" + "=" * 60)
print("t-SNE VISUALIZATION")
print("=" * 60)

STATE_NAMES = {0: "Normal", 1: "Precursor", 2: "Threat", 3: "Post-Threat"}
STATE_COLORS = {0: "#6baed6", 1: "#fd8d3c", 2: "#e31a1c", 3: "#31a354"}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("t-SNE: Hidden State Representations\nColor = True State Type", fontsize=14)

for ax_idx, config_name in enumerate(["D", "E"]):
    # Use seed 42 for visualization
    seed = EXP_CFG.n_seeds[0]
    fin = final_metrics_all[config_name][seed]
    hidden_states = np.array(fin["hidden_states"])
    true_labels = np.array(fin["true_labels"])

    # Subsample to 2000 for speed
    n_vis = min(2000, len(hidden_states))
    idx = np.random.choice(len(hidden_states), n_vis, replace=False)
    H = hidden_states[idx]
    L = true_labels[idx]

    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)

    print(f"  Running t-SNE for Config {config_name} ({n_vis} samples)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, verbose=0)
    H_2d = tsne.fit_transform(H_scaled)

    ax = axes[ax_idx]
    for label_val, label_name in STATE_NAMES.items():
        mask = L == label_val
        if mask.sum() > 0:
            ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                       c=STATE_COLORS[label_val], label=label_name,
                       s=8, alpha=0.6)
    ax.set_title(f"Config {config_name} (seed {seed})", fontsize=12)
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

plt.tight_layout()
plt.savefig("./plots/tsne_hidden_states.png", dpi=150, bbox_inches='tight')
plt.close()
print("t-SNE plot saved to ./plots/tsne_hidden_states.png")

# %% ─── CELL 11 — LEARNING CURVES PLOT ───────────────────────────────────────

print("\n" + "=" * 60)
print("LEARNING CURVES")
print("=" * 60)

def smooth(arr, window=10):
    return np.convolve(arr, np.ones(window) / window, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 2 Learning Curves — Config D vs Config E\n(5 seeds, shaded = ±1 std)", fontsize=13)

plot_metrics = [
    ("precursor_det_rate", "Precursor Detection Rate", axes[0, 0]),
    ("threat_det_rate", "Threat Detection Rate", axes[0, 1]),
    ("surprise_rate", "Surprise Event Rate", axes[1, 0]),
    ("nonzero_reward_rate", "Non-Zero Reward Rate", axes[1, 1]),
]
colors = {"D": "#2196F3", "E": "#FF5722"}
window = 10

for metric, title, ax in plot_metrics:
    for config_name in ["D", "E"]:
        all_seed_curves = []
        for seed in EXP_CFG.n_seeds:
            curve = [e[metric] for e in all_results[config_name][seed]]
            all_seed_curves.append(smooth(curve, window))
        min_len = min(len(c) for c in all_seed_curves)
        arr = np.array([c[:min_len] for c in all_seed_curves])
        mean = arr.mean(0)
        std = arr.std(0)
        x = np.arange(min_len)
        ax.plot(x, mean, color=colors[config_name], label=f"Config {config_name}", linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=colors[config_name])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Episode (smoothed)")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./plots/learning_curves_phase2.png", dpi=150, bbox_inches='tight')
plt.close()
print("Learning curves saved to ./plots/learning_curves_phase2.png")

# %% ─── CELL 12 — SURPRISE COMPOSITION BAR CHART ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Surprise Event Composition by True State\n(all seeds, all episodes)", fontsize=13)

label_map = {0: "Normal", 1: "Precursor", 2: "Threat", 3: "Post-Threat"}
bar_colors = [STATE_COLORS[k] for k in range(4)]

for ax_idx, config_name in enumerate(["D", "E"]):
    counts = [0, 0, 0, 0]
    for seed in EXP_CFG.n_seeds:
        for ev in surprise_traces_output[config_name][str(seed)]:
            lbl = ev["true_label"]
            if 0 <= lbl <= 3:
                counts[lbl] += 1
    total = sum(counts)
    pcts = [c / max(total, 1) * 100 for c in counts]
    axes[ax_idx].bar(label_map.values(), pcts, color=bar_colors)
    axes[ax_idx].axhline(y=35, color='red', linestyle='--', linewidth=1.5, label='Target (35%)')
    axes[ax_idx].set_title(f"Config {config_name}", fontsize=12)
    axes[ax_idx].set_ylabel("% of surprise events")
    axes[ax_idx].set_ylim(0, 100)
    axes[ax_idx].legend()
    for i, (bar_label, pct) in enumerate(zip(label_map.values(), pcts)):
        axes[ax_idx].text(i, pct + 1, f"{pct:.1f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("./plots/surprise_composition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Surprise composition chart saved to ./plots/surprise_composition.png")

# %% ─── CELL 13 — SUMMARY & DOWNLOAD CHECKLIST ───────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2 SUMMARY")
print("=" * 60)

print("\nStatistical Results:")
for metric, res in stat_results.items():
    sig_str = "SIGNIFICANT" if res["significant"] else "not significant"
    print(f"  {metric}: D={res['D_mean']:.4f} E={res['E_mean']:.4f}  p={res['p_val']:.4f} ({sig_str})")

print("\nLinear Probe Results:")
for config_name, res in probe_results.items():
    print(f"  Config {config_name}: mean accuracy = {res['mean_acc']:.3f} (chance = 0.25)")

print("\nFiles to download before session ends:")
print("  ./surprise_traces_phase2.json  ← REQUIRED for Phase 3")
print("  ./checkpoints/                 ← model weights")
print("  ./plots/                       ← all visualizations")
print("  ./results/                     ← statistical outputs")

print("\nPhase 2 success criteria check:")
prec_p = stat_results["precursor_det_rate"]["p_val"]
prec_e_mean = stat_results["precursor_det_rate"]["E_mean"]
prec_d_mean = stat_results["precursor_det_rate"]["D_mean"]
probe_d_acc = probe_results["D"]["mean_acc"]

c1 = prec_p < 0.05 and prec_e_mean > prec_d_mean
c2 = probe_d_acc > 0.50
# c3 and c4 require manual inspection of plots
print(f"  □ E > D on precursor detection (p<0.05):  {'✓' if c1 else '✗'}")
print(f"  □ Linear probe D accuracy > 50%:           {'✓' if c2 else '✗'}")
print(f"  □ Surprise traces >35% precursor:          [check surprise_composition.png]")
print(f"  □ t-SNE shows clustering:                  [check tsne_hidden_states.png]")
if c1 or c2:
    print("\n[PHASE 2 PASSED] At least one success criterion met. Proceed to Phase 3.")
else:
    print("\n[PHASE 2 PARTIAL] Check plots. If probe <0.45, extend to N_EPS=300.")
