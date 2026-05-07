"""
cpu_baseline_experiment.py
GFP Research — v1 CPU Baseline Pilot
Necessity Labs / TrialBlazer23

Runs 2 seeds × 4 configs × 30 episodes × 600 steps on CPU.
No GPU required.

Usage:
    python cpu_baseline_experiment.py

Outputs:
    - Console results table
    - results_v1.json
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    """Defines the signal environment parameters."""
    # State machine
    normal_duration_min: int = 20
    normal_duration_max: int = 60
    precursor_duration_min: int = 5
    precursor_duration_max: int = 15
    threat_duration_min: int = 8
    threat_duration_max: int = 20

    # Signal statistics
    obs_dim: int = 8
    base_std: float = 1.0
    precursor_mean_shift: float = 0.08   # sub-threshold; hard to detect
    threat_mean_shift: float = 0.35      # detectable but not obvious

    # Transitions
    normal_to_precursor_prob: float = 0.03  # per step while in normal
    precursor_to_threat_prob: float = 0.10  # per step while in precursor
    threat_to_normal_prob: float = 0.15     # per step while in threat

    # Episode
    episode_steps: int = 600


@dataclass
class AgentConfig:
    """Defines the agent hyperparameters."""
    hidden_size: int = 64
    num_layers: int = 1
    lr: float = 3e-4
    gamma: float = 0.99

    # Config B / C
    ward_correct_reward: float = 1.0
    false_alarm_penalty: float = -0.5

    # Config C / D — intrinsic surprise
    surprise_bonus_scale: float = 0.5
    surprise_threshold_std: float = 1.0   # mean + N*std to gate reward

    # Config A — BCE
    pos_weight: float = 5.0              # upweight threat class in BCE


# ---------------------------------------------------------------------------
# Signal Environment
# ---------------------------------------------------------------------------

# State constants
STATE_NORMAL = 0
STATE_PRECURSOR = 1
STATE_THREAT = 2


class SignalEnvironment:
    """
    1-D signal environment with three hidden states: normal, precursor, threat.

    The agent observes a noisy obs_dim-dimensional signal. The mean shifts
    slightly during precursor (0.08) and more during threat (0.35).
    The agent is NOT told the state — it must infer from signal statistics.

    Action space: binary {0=no_action, 1=alert}.
    """

    def __init__(self, cfg: EnvironmentConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = STATE_NORMAL
        self.steps_in_state = 0
        self.state_duration = self._sample_duration(STATE_NORMAL)
        self.step_count = 0
        self.ward_damage = 0.0
        self.ward_health = 100.0
        return self._observe()

    def _sample_duration(self, state: int) -> int:
        cfg = self.cfg
        if state == STATE_NORMAL:
            return int(self.rng.integers(cfg.normal_duration_min, cfg.normal_duration_max + 1))
        elif state == STATE_PRECURSOR:
            return int(self.rng.integers(cfg.precursor_duration_min, cfg.precursor_duration_max + 1))
        else:
            return int(self.rng.integers(cfg.threat_duration_min, cfg.threat_duration_max + 1))

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        if self.state == STATE_NORMAL:
            mean = 0.0
        elif self.state == STATE_PRECURSOR:
            mean = cfg.precursor_mean_shift
        else:
            mean = cfg.threat_mean_shift
        obs = self.rng.normal(mean, cfg.base_std, size=cfg.obs_dim)
        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, dict]:
        """
        Returns (next_obs, info_dict).
        info dict contains: state, prev_state, ward_health, threat_hit, false_alarm.
        """
        prev_state = self.state
        self.steps_in_state += 1
        self.step_count += 1

        # State transitions
        if self.state == STATE_NORMAL:
            if self.rng.random() < self.cfg.normal_to_precursor_prob:
                self.state = STATE_PRECURSOR
                self.steps_in_state = 0
                self.state_duration = self._sample_duration(STATE_PRECURSOR)
        elif self.state == STATE_PRECURSOR:
            if self.rng.random() < self.cfg.precursor_to_threat_prob:
                self.state = STATE_THREAT
                self.steps_in_state = 0
                self.state_duration = self._sample_duration(STATE_THREAT)
            elif self.steps_in_state >= self.state_duration:
                # precursor resolved without threat
                self.state = STATE_NORMAL
                self.steps_in_state = 0
                self.state_duration = self._sample_duration(STATE_NORMAL)
        else:  # THREAT
            # Ward takes damage while in threat state and no action taken
            if action == 0:
                self.ward_health = max(0.0, self.ward_health - 2.0)
            if self.rng.random() < self.cfg.threat_to_normal_prob:
                self.state = STATE_NORMAL
                self.steps_in_state = 0
                self.state_duration = self._sample_duration(STATE_NORMAL)

        obs = self._observe()

        threat_hit = (prev_state == STATE_THREAT and action == 1)
        false_alarm = (prev_state != STATE_THREAT and action == 1)

        info = {
            "state": prev_state,
            "next_state": self.state,
            "ward_health": self.ward_health,
            "threat_hit": threat_hit,
            "false_alarm": false_alarm,
            "is_precursor": (prev_state == STATE_PRECURSOR),
            "is_threat": (prev_state == STATE_THREAT),
        }
        return obs, info


# ---------------------------------------------------------------------------
# Welford running statistics (online mean + variance)
# ---------------------------------------------------------------------------

class WelfordStats:
    """Tracks running mean and variance using Welford's algorithm."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 1.0
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 1e-8))

    def normalized(self, x: float) -> float:
        return (x - self.mean) / max(self.std, 1e-8)


# ---------------------------------------------------------------------------
# LSTM backbone
# ---------------------------------------------------------------------------

class LSTMBackbone(nn.Module):
    """
    Shared LSTM backbone used by all configs.
    Input: obs_dim
    Output: hidden state of shape (hidden_size,)
    Also has:
      - pred_head: predicts next observation (for computing prediction error)
      - policy_head: outputs logit for binary action
    """

    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.pred_head = nn.Linear(hidden_size, obs_dim)
        self.policy_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        obs: torch.Tensor,           # (1, 1, obs_dim)
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Returns: (hidden, pred_next_obs, policy_logit, new_hx)
        """
        out, new_hx = self.lstm(obs, hx)          # out: (1, 1, hidden_size)
        hidden = out.squeeze(0).squeeze(0)         # (hidden_size,)
        pred = self.pred_head(hidden)              # (obs_dim,)
        logit = self.policy_head(hidden)           # (1,)
        return hidden, pred, logit, new_hx


# ---------------------------------------------------------------------------
# Helper: compute discounted returns
# ---------------------------------------------------------------------------

def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# ---------------------------------------------------------------------------
# Config A — Supervised BCE (withheld precursor label)
# ---------------------------------------------------------------------------

class ConfigA_Agent:
    """
    Supervised binary cross-entropy classifier.
    Labels: 0 = normal (includes precursor — agent never told precursor exists)
            1 = threat
    Uses pos_weight to handle class imbalance.
    Action is deterministic: act if sigmoid(logit) > 0.5.
    """

    def __init__(self, env_cfg: EnvironmentConfig, agent_cfg: AgentConfig):
        self.cfg = agent_cfg
        self.net = LSTMBackbone(env_cfg.obs_dim, agent_cfg.hidden_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=agent_cfg.lr)
        pos_w = torch.tensor([agent_cfg.pos_weight])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.hx = None
        self.reset_hidden()

    def reset_hidden(self):
        h = torch.zeros(1, 1, self.cfg.hidden_size)
        c = torch.zeros(1, 1, self.cfg.hidden_size)
        self.hx = (h, c)

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, dict]:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, _, logit, self.hx = self.net(obs_t, self.hx)
        prob = torch.sigmoid(logit).item()
        action = int(prob > 0.5)
        return action, {"logit": logit.item(), "prob": prob}

    def update(self, obs: np.ndarray, action: int, reward: float, info: dict):
        """Supervised step — ignores reward, trains on threat label."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Label: threat=1, everything else=0 (precursor is masked as normal)
        label = torch.tensor([[float(info["is_threat"])]])
        _, _, logit, _ = self.net(obs_t, self.hx)
        loss = self.criterion(logit.unsqueeze(0), label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ---------------------------------------------------------------------------
# Config B — REINFORCE with ward health reward
# ---------------------------------------------------------------------------

class ConfigB_Agent:
    """
    Policy gradient (REINFORCE).
    Reward: +ward_correct_reward for action=1 when threat, -false_alarm_penalty for action=1 when not threat.
    """

    def __init__(self, env_cfg: EnvironmentConfig, agent_cfg: AgentConfig):
        self.cfg = agent_cfg
        self.net = LSTMBackbone(env_cfg.obs_dim, agent_cfg.hidden_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=agent_cfg.lr)
        self.hx = None
        self.reset_hidden()
        # Episode buffers
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def reset_hidden(self):
        h = torch.zeros(1, 1, self.cfg.hidden_size)
        c = torch.zeros(1, 1, self.cfg.hidden_size)
        self.hx = (h, c)

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, dict]:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        _, _, logit, self.hx = self.net(obs_t, self.hx)
        prob = torch.sigmoid(logit)
        dist = torch.distributions.Bernoulli(prob)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        self.log_probs.append(log_prob)
        return int(action_t.item()), {"log_prob": log_prob.item()}

    def collect_reward(self, reward: float):
        self.rewards.append(reward)

    def compute_reward(self, info: dict, action: int) -> float:
        if action == 1 and info["is_threat"]:
            return self.cfg.ward_correct_reward
        elif action == 1 and not info["is_threat"]:
            return self.cfg.false_alarm_penalty
        return 0.0

    def end_episode(self):
        if not self.rewards:
            return
        returns = compute_returns(self.rewards, self.cfg.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.stack([-lp * R for lp, R in zip(self.log_probs, returns_t)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []


# ---------------------------------------------------------------------------
# Config C — REINFORCE + intrinsic surprise bonus
# ---------------------------------------------------------------------------

class ConfigC_Agent(ConfigB_Agent):
    """
    Config B + intrinsic surprise bonus added to extrinsic reward.
    Surprise = normalized prediction error (via Welford stats).
    Bonus is gated: only given if pred_error > mean + threshold_std * std.
    """

    def __init__(self, env_cfg: EnvironmentConfig, agent_cfg: AgentConfig):
        super().__init__(env_cfg, agent_cfg)
        self.pred_error_stats = WelfordStats()
        self.prev_obs: Optional[np.ndarray] = None

    def reset_hidden(self):
        super().reset_hidden()
        self.prev_obs = None

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, dict]:
        # Compute prediction error on current obs vs. what we predicted last step
        action, act_info = super().act(obs, info)
        # Get prediction for current hidden state
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, pred, _, _ = self.net(obs_t, self.hx)
        act_info["pred"] = pred.detach().numpy()
        return action, act_info

    def compute_reward(self, info: dict, action: int) -> float:
        base = super().compute_reward(info, action)
        # Compute surprise from previous prediction vs current obs
        surprise_bonus = 0.0
        if self.prev_obs is not None:
            obs_t = torch.tensor(self.prev_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                _, pred, _, _ = self.net(obs_t, self.hx)
            # This is a simplified pred error using stored prev obs
            # In full implementation we'd store predicted next obs
        return base + surprise_bonus

    def compute_reward_with_pred_error(self, info: dict, action: int, pred_error: float) -> float:
        base = super().compute_reward(info, action)
        self.pred_error_stats.update(pred_error)
        threshold = self.pred_error_stats.mean + self.cfg.surprise_threshold_std * self.pred_error_stats.std
        if pred_error > threshold:
            normalized = self.pred_error_stats.normalized(pred_error)
            surprise_bonus = self.cfg.surprise_bonus_scale * max(0.0, normalized)
        else:
            surprise_bonus = 0.0
        return base + surprise_bonus


# ---------------------------------------------------------------------------
# Config D — Intrinsic curiosity only (surprise-gated)
# ---------------------------------------------------------------------------

class ConfigD_Agent:
    """
    Pure curiosity agent. No extrinsic reward, no ward health signal, no threat label.
    Reward = 0.5 * normalized_pred_error  IFF pred_error > running_mean + 1*std AND action==1.
    REINFORCE with this intrinsic reward only.
    """

    def __init__(self, env_cfg: EnvironmentConfig, agent_cfg: AgentConfig):
        self.cfg = agent_cfg
        self.net = LSTMBackbone(env_cfg.obs_dim, agent_cfg.hidden_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=agent_cfg.lr)
        self.hx = None
        self.pred_error_stats = WelfordStats()
        self.reset_hidden()
        # Episode buffers
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        # Prediction tracking
        self.last_pred: Optional[torch.Tensor] = None

    def reset_hidden(self):
        h = torch.zeros(1, 1, self.cfg.hidden_size)
        c = torch.zeros(1, 1, self.cfg.hidden_size)
        self.hx = (h, c)
        self.last_pred = None

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, dict]:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Compute pred error against last prediction
        pred_error = 0.0
        if self.last_pred is not None:
            with torch.no_grad():
                pred_error = F.mse_loss(
                    self.last_pred,
                    obs_t.squeeze(0).squeeze(0)
                ).item()

        # Forward pass
        _, pred, logit, self.hx = self.net(obs_t, self.hx)
        self.last_pred = pred.detach()

        # Policy
        prob = torch.sigmoid(logit)
        dist = torch.distributions.Bernoulli(prob)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        self.log_probs.append(log_prob)

        action = int(action_t.item())

        # Compute intrinsic reward
        self.pred_error_stats.update(pred_error)
        threshold = (
            self.pred_error_stats.mean
            + self.cfg.surprise_threshold_std * self.pred_error_stats.std
        )
        if pred_error > threshold and action == 1:
            normalized = self.pred_error_stats.normalized(pred_error)
            reward = self.cfg.surprise_bonus_scale * max(0.0, normalized)
        else:
            reward = 0.0

        self.rewards.append(reward)

        return action, {"pred_error": pred_error, "intrinsic_reward": reward}

    def end_episode(self):
        if not self.rewards:
            return
        returns = compute_returns(self.rewards, self.cfg.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.stack([-lp * R for lp, R in zip(self.log_probs, returns_t)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode_A(agent: ConfigA_Agent, env: SignalEnvironment, train: bool = True) -> dict:
    obs = env.reset()
    agent.reset_hidden()
    metrics = {"prec_det": [], "threat_det": [], "fp": []}

    for _ in range(env.cfg.episode_steps):
        action, act_info = agent.act(obs, {})
        next_obs, info = env.step(action)

        if train:
            agent.update(obs, action, 0.0, info)

        # Track metrics
        if info["is_precursor"]:
            metrics["prec_det"].append(float(action))
        if info["is_threat"]:
            metrics["threat_det"].append(float(action))
        if not info["is_threat"] and not info["is_precursor"]:
            metrics["fp"].append(float(action))

        obs = next_obs

    return {
        "ward_health": env.ward_health,
        "prec_det": float(np.mean(metrics["prec_det"])) if metrics["prec_det"] else 0.0,
        "threat_det": float(np.mean(metrics["threat_det"])) if metrics["threat_det"] else 0.0,
        "fp": float(np.mean(metrics["fp"])) if metrics["fp"] else 0.0,
    }


def run_episode_BC(agent: ConfigB_Agent, env: SignalEnvironment, train: bool = True) -> dict:
    obs = env.reset()
    agent.reset_hidden()
    agent.log_probs = []
    agent.rewards = []
    metrics = {"prec_det": [], "threat_det": [], "fp": []}

    if isinstance(agent, ConfigC_Agent):
        # For C, we track prediction error for surprise
        prev_obs = None
        pred_error_stats = WelfordStats()

    for _ in range(env.cfg.episode_steps):
        action, act_info = agent.act(obs, {})
        next_obs, info = env.step(action)

        if isinstance(agent, ConfigC_Agent):
            # Compute surprise from prediction error
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                _, pred, _, _ = agent.net(obs_t, agent.hx)
            next_t = torch.tensor(next_obs, dtype=torch.float32)
            pred_error = F.mse_loss(pred.squeeze(), next_t).item()
            reward = agent.compute_reward_with_pred_error(info, action, pred_error)
        else:
            reward = agent.compute_reward(info, action)

        agent.collect_reward(reward)

        if info["is_precursor"]:
            metrics["prec_det"].append(float(action))
        if info["is_threat"]:
            metrics["threat_det"].append(float(action))
        if not info["is_threat"] and not info["is_precursor"]:
            metrics["fp"].append(float(action))

        obs = next_obs

    if train:
        agent.end_episode()

    return {
        "ward_health": env.ward_health,
        "prec_det": float(np.mean(metrics["prec_det"])) if metrics["prec_det"] else 0.0,
        "threat_det": float(np.mean(metrics["threat_det"])) if metrics["threat_det"] else 0.0,
        "fp": float(np.mean(metrics["fp"])) if metrics["fp"] else 0.0,
    }


def run_episode_D(agent: ConfigD_Agent, env: SignalEnvironment, train: bool = True) -> dict:
    obs = env.reset()
    agent.reset_hidden()
    agent.log_probs = []
    agent.rewards = []
    metrics = {"prec_det": [], "threat_det": [], "fp": []}

    for _ in range(env.cfg.episode_steps):
        action, act_info = agent.act(obs, {})
        next_obs, info = env.step(action)

        if info["is_precursor"]:
            metrics["prec_det"].append(float(action))
        if info["is_threat"]:
            metrics["threat_det"].append(float(action))
        if not info["is_threat"] and not info["is_precursor"]:
            metrics["fp"].append(float(action))

        obs = next_obs

    if train:
        agent.end_episode()

    return {
        "ward_health": env.ward_health,
        "prec_det": float(np.mean(metrics["prec_det"])) if metrics["prec_det"] else 0.0,
        "threat_det": float(np.mean(metrics["threat_det"])) if metrics["threat_det"] else 0.0,
        "fp": float(np.mean(metrics["fp"])) if metrics["fp"] else 0.0,
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment():
    seeds = [42, 7]
    n_episodes = 30
    env_cfg = EnvironmentConfig()
    agent_cfg = AgentConfig()

    all_results = {}

    config_names = {
        "A": "Supervised BCE (withheld precursor)",
        "B": "REINFORCE + ward health",
        "C": "REINFORCE + ward health + intrinsic",
        "D": "Intrinsic curiosity only",
    }

    for seed in seeds:
        set_seed(seed)
        all_results[str(seed)] = {}
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        for config_key in ["A", "B", "C", "D"]:
            print(f"\n  Config {config_key}: {config_names[config_key]}")
            env = SignalEnvironment(env_cfg, seed=seed)

            if config_key == "A":
                agent = ConfigA_Agent(env_cfg, agent_cfg)
            elif config_key == "B":
                agent = ConfigB_Agent(env_cfg, agent_cfg)
            elif config_key == "C":
                agent = ConfigC_Agent(env_cfg, agent_cfg)
            else:
                agent = ConfigD_Agent(env_cfg, agent_cfg)

            episode_metrics = []
            for ep in range(n_episodes):
                if config_key == "A":
                    ep_result = run_episode_A(agent, env, train=True)
                elif config_key in ("B", "C"):
                    ep_result = run_episode_BC(agent, env, train=True)
                else:
                    ep_result = run_episode_D(agent, env, train=True)

                episode_metrics.append(ep_result)

                if (ep + 1) % 10 == 0:
                    recent = episode_metrics[-10:]
                    avg_prec = np.mean([m["prec_det"] for m in recent])
                    avg_ward = np.mean([m["ward_health"] for m in recent])
                    print(
                        f"    ep {ep+1:3d} | ward={avg_ward:6.1f} | "
                        f"prec_det={avg_prec:.3f}"
                    )

            # Aggregate last 10 episodes (or all if fewer)
            last_n = episode_metrics[-10:]
            summary = {
                "ward_health": float(np.mean([m["ward_health"] for m in last_n])),
                "prec_det": float(np.mean([m["prec_det"] for m in last_n])),
                "threat_det": float(np.mean([m["threat_det"] for m in last_n])),
                "fp": float(np.mean([m["fp"] for m in last_n])),
                "all_episodes": episode_metrics,
            }
            all_results[str(seed)][config_key] = summary

    # ---------------------------------------------------------------------------
    # Aggregate across seeds
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("RESULTS — Averaged across seeds")
    print(f"{'='*60}")
    print(f"\n{'Config':<8} {'Description':<35} {'ward_health':>12} {'prec_det':>10} {'threat_det':>11} {'fp':>8}")
    print("-" * 88)

    aggregated = {}
    for config_key, desc in config_names.items():
        seed_results = [all_results[str(s)][config_key] for s in seeds]
        avg = {
            k: float(np.mean([r[k] for r in seed_results]))
            for k in ["ward_health", "prec_det", "threat_det", "fp"]
        }
        aggregated[config_key] = avg
        print(
            f"{'Config '+config_key:<8} {desc:<35} "
            f"{avg['ward_health']:>12.1f} {avg['prec_det']:>10.3f} "
            f"{avg['threat_det']:>11.3f} {avg['fp']:>8.3f}"
        )

    print()

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    output = {
        "version": "v1_cpu_baseline",
        "date": "2026-04-09",
        "seeds": seeds,
        "n_episodes": n_episodes,
        "episode_steps": env_cfg.episode_steps,
        "hidden_size": agent_cfg.hidden_size,
        "env_config": {
            "obs_dim": env_cfg.obs_dim,
            "base_std": env_cfg.base_std,
            "precursor_mean_shift": env_cfg.precursor_mean_shift,
            "threat_mean_shift": env_cfg.threat_mean_shift,
        },
        "per_seed": all_results,
        "aggregated": aggregated,
    }

    out_path = "results_v1.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
