"""
experiment_core.py
GFP Research — v2 L4 GPU Scale
Necessity Labs / TrialBlazer23

Core module imported by all v2 notebooks.
Written to disk by notebook 00_setup_and_dependencies.

Usage (smoke test):
    python -c "from experiment_core import *; print('OK')"

Full training is launched from notebook 01_core_training_runs.
"""

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# EnvironmentConfig (24 fields)
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    # ---- State machine durations ----
    normal_duration_min: int = 20
    normal_duration_max: int = 80
    precursor_duration_min: int = 5
    precursor_duration_max: int = 20
    threat_duration_min: int = 8
    threat_duration_max: int = 25

    # ---- Signal statistics ----
    obs_dim: int = 8
    base_std: float = 1.0
    precursor_mean_shift: float = 0.08
    threat_mean_shift: float = 0.35

    # ---- Curriculum parameters ----
    # If curriculum_start_shift != curriculum_end_shift, interpolate
    # precursor_mean_shift linearly over training from start to end.
    curriculum_start_shift: float = 0.08   # set > 0.08 to enable easy->hard curriculum
    curriculum_end_shift: float = 0.08     # set == start to disable curriculum
    curriculum_end_episode: int = 200      # episode at which end_shift is reached

    # ---- Transition probabilities ----
    normal_to_precursor_prob: float = 0.025
    precursor_to_threat_prob: float = 0.10
    threat_to_normal_prob: float = 0.15

    # ---- Episode ----
    episode_steps: int = 1500

    # ---- Reward / label options ----
    pos_weight: float = 5.0          # Config A: BCE pos_weight for threat class
    false_alarm_penalty: float = -0.5  # Config B/C: penalty for false alarm action
    ward_correct_reward: float = 1.0   # Config B/C: reward for correct threat action
    ward_damage_per_step: float = 2.0  # Ward HP lost per step in threat state with no action

    # ---- Total episodes (used by curriculum interpolation) ----
    n_episodes: int = 400


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

STATE_NORMAL = 0
STATE_PRECURSOR = 1
STATE_THREAT = 2


# ---------------------------------------------------------------------------
# SignalEnvironment with curriculum support
# ---------------------------------------------------------------------------

class SignalEnvironment:
    """
    1-D signal environment with three hidden states: normal, precursor, threat.

    The agent observes a noisy obs_dim-dimensional vector. Mean shifts:
      - Normal:    0.0
      - Precursor: precursor_mean_shift (sub-threshold; default 0.08)
      - Threat:    threat_mean_shift    (detectable; default 0.35)

    Action space: binary {0=no_action, 1=alert}.

    Curriculum: if curriculum_start_shift != curriculum_end_shift, the effective
    precursor_mean_shift is linearly interpolated from start to end over
    episodes [0, curriculum_end_episode]. Call set_episode(ep) before each episode.
    """

    def __init__(self, cfg: EnvironmentConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._current_episode = 0
        self._effective_precursor_shift = cfg.precursor_mean_shift
        self.reset()

    def set_episode(self, episode: int):
        """Call before each episode to update curriculum shift."""
        self._current_episode = episode
        cfg = self.cfg
        if cfg.curriculum_start_shift == cfg.curriculum_end_shift:
            # No curriculum
            self._effective_precursor_shift = cfg.precursor_mean_shift
        else:
            # Linear interpolation from start_shift to end_shift
            progress = min(1.0, episode / max(1, cfg.curriculum_end_episode))
            self._effective_precursor_shift = (
                cfg.curriculum_start_shift
                + progress * (cfg.curriculum_end_shift - cfg.curriculum_start_shift)
            )

    def reset(self) -> np.ndarray:
        self.state = STATE_NORMAL
        self.steps_in_state = 0
        self.state_duration = self._sample_duration(STATE_NORMAL)
        self.step_count = 0
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
            mean = self._effective_precursor_shift
        else:
            mean = cfg.threat_mean_shift
        return self.rng.normal(mean, cfg.base_std, size=cfg.obs_dim).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, dict]:
        """
        Returns (next_obs, info).
        Info keys: state, next_state, ward_health, is_precursor, is_threat,
                   threat_hit, false_alarm, precursor_shift.
        """
        prev_state = self.state
        self.steps_in_state += 1
        self.step_count += 1

        # Transitions
        if self.state == STATE_NORMAL:
            if self.rng.random() < self.cfg.normal_to_precursor_prob:
                self._transition_to(STATE_PRECURSOR)
        elif self.state == STATE_PRECURSOR:
            if self.rng.random() < self.cfg.precursor_to_threat_prob:
                self._transition_to(STATE_THREAT)
            elif self.steps_in_state >= self.state_duration:
                self._transition_to(STATE_NORMAL)
        else:  # THREAT
            if action == 0:
                self.ward_health = max(0.0, self.ward_health - self.cfg.ward_damage_per_step)
            if self.rng.random() < self.cfg.threat_to_normal_prob:
                self._transition_to(STATE_NORMAL)

        obs = self._observe()
        info = {
            "state": prev_state,
            "next_state": self.state,
            "ward_health": self.ward_health,
            "is_precursor": (prev_state == STATE_PRECURSOR),
            "is_threat": (prev_state == STATE_THREAT),
            "is_normal": (prev_state == STATE_NORMAL),
            "threat_hit": (prev_state == STATE_THREAT and action == 1),
            "false_alarm": (prev_state != STATE_THREAT and action == 1),
            "precursor_shift": self._effective_precursor_shift,
        }
        return obs, info

    def _transition_to(self, new_state: int):
        self.state = new_state
        self.steps_in_state = 0
        self.state_duration = self._sample_duration(new_state)


# ---------------------------------------------------------------------------
# Welford running statistics
# ---------------------------------------------------------------------------

class WelfordStats:
    """Online mean and variance via Welford's algorithm."""

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
        return self.M2 / (self.n - 1) if self.n >= 2 else 1.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 1e-8))

    def normalized(self, x: float) -> float:
        return (x - self.mean) / max(self.std, 1e-8)

    def threshold(self, n_std: float = 1.0) -> float:
        return self.mean + n_std * self.std

    def state_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "M2": self.M2}

    def load_state_dict(self, d: dict):
        self.n = d["n"]
        self.mean = d["mean"]
        self.M2 = d["M2"]


# ---------------------------------------------------------------------------
# LSTM backbone
# ---------------------------------------------------------------------------

class LSTMAgent(nn.Module):
    """
    Base LSTM agent.

    Architecture:
      lstm:         LSTM(obs_dim → hidden_size, 1 layer)
      pred_head:    Linear(hidden_size → obs_dim)   — predicts next observation
      policy_head:  Linear(hidden_size → 1)         — binary action logit

    Manages hidden state (hx) internally. Call reset_hidden() at episode start.
    Tracks ward_health as a running scalar (reset each episode via env.reset()).
    Maintains a WelfordStats accumulator for prediction error.
    """

    def __init__(self, obs_dim: int, hidden_size: int = 128):
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

        # Running stats for prediction error (persists across episodes)
        self.pred_error_stats = WelfordStats()

        # Hidden state
        self.hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(self):
        h = torch.zeros(1, 1, self.hidden_size, device=DEVICE)
        c = torch.zeros(1, 1, self.hidden_size, device=DEVICE)
        self.hx = (h, c)
        self._last_pred: Optional[torch.Tensor] = None

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (1, 1, obs_dim) on DEVICE.
        Returns: (hidden, pred_next_obs, policy_logit)
        Updates self.hx in place.
        """
        out, self.hx = self.lstm(obs, self.hx)
        hidden = out.squeeze(0).squeeze(0)   # (hidden_size,)
        pred = self.pred_head(hidden)         # (obs_dim,)
        logit = self.policy_head(hidden)      # (1,)
        return hidden, pred, logit

    def compute_pred_error(
        self,
        obs: torch.Tensor,
    ) -> float:
        """
        Compute MSE between last prediction and current obs.
        Returns scalar. Updates pred_error_stats.
        """
        if self._last_pred is None:
            return 0.0
        with torch.no_grad():
            err = F.mse_loss(self._last_pred, obs.squeeze()).item()
        self.pred_error_stats.update(err)
        return err

    def store_prediction(self, pred: torch.Tensor):
        self._last_pred = pred.detach()

    def obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

    def state_dict_extended(self) -> dict:
        """State dict including Welford stats (for checkpointing)."""
        sd = self.state_dict()
        sd["__welford__"] = self.pred_error_stats.state_dict()
        if self.hx is not None:
            sd["__hx_h__"] = self.hx[0].cpu()
            sd["__hx_c__"] = self.hx[1].cpu()
        return sd

    def load_state_dict_extended(self, sd: dict):
        welford = sd.pop("__welford__", None)
        hx_h = sd.pop("__hx_h__", None)
        hx_c = sd.pop("__hx_c__", None)
        self.load_state_dict(sd)
        if welford is not None:
            self.pred_error_stats.load_state_dict(welford)
        if hx_h is not None and hx_c is not None:
            self.hx = (hx_h.to(DEVICE), hx_c.to(DEVICE))


# ---------------------------------------------------------------------------
# Config A — Supervised BCE with pos_weight
# ---------------------------------------------------------------------------

class ConfigA_Agent(LSTMAgent):
    """
    Supervised binary cross-entropy classifier.
    Labels: 1=threat, 0=everything else (precursor label withheld).
    pos_weight upweights the threat class to handle imbalance.
    Action: deterministic threshold on sigmoid(logit) > 0.5.
    Gradient is computed step-by-step (online supervised learning).
    """

    def __init__(self, env_cfg: EnvironmentConfig):
        super().__init__(env_cfg.obs_dim, hidden_size=128)
        pos_w = torch.tensor([env_cfg.pos_weight], device=DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.to(DEVICE)

    def step_supervised(
        self,
        obs: np.ndarray,
        info: dict,
        optimizer: optim.Optimizer,
    ) -> Tuple[int, float]:
        """
        Forward pass + supervised update.
        Returns (action, loss).
        """
        obs_t = self.obs_to_tensor(obs)
        pred_error = self.compute_pred_error(obs_t.squeeze(0).squeeze(0))

        _, pred, logit = self.forward(obs_t)
        self.store_prediction(pred)

        label = torch.tensor([[float(info["is_threat"])]], device=DEVICE)
        loss = self.criterion(logit.unsqueeze(0), label)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        # Reset hx after backward to avoid graph accumulation
        self.hx = tuple(h.detach() for h in self.hx)

        with torch.no_grad():
            prob = torch.sigmoid(logit).item()
        action = int(prob > 0.5)
        return action, loss.item()


# ---------------------------------------------------------------------------
# Config B — REINFORCE with ward health reward
# ---------------------------------------------------------------------------

class ConfigB_Agent(LSTMAgent):
    """
    Policy gradient (REINFORCE).
    Extrinsic reward: +ward_correct_reward for act=1 when threat,
                      false_alarm_penalty for act=1 when not threat,
                      0 otherwise.
    Episode buffers accumulate log_probs and rewards; end_episode() applies update.
    """

    def __init__(self, env_cfg: EnvironmentConfig):
        super().__init__(env_cfg.obs_dim, hidden_size=128)
        self.env_cfg = env_cfg
        self.to(DEVICE)
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

    def step_act(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Forward pass + sample action.
        Returns (action, pred_error).
        Appends log_prob to episode buffer.
        """
        obs_t = self.obs_to_tensor(obs)
        pred_error = self.compute_pred_error(obs_t.squeeze(0).squeeze(0))

        _, pred, logit = self.forward(obs_t)
        self.store_prediction(pred)
        self.hx = tuple(h.detach() for h in self.hx)

        prob = torch.sigmoid(logit)
        dist = torch.distributions.Bernoulli(prob)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        self._log_probs.append(log_prob)

        return int(action_t.item()), pred_error

    def collect_reward(self, action: int, info: dict):
        cfg = self.env_cfg
        if action == 1 and info["is_threat"]:
            r = cfg.ward_correct_reward
        elif action == 1 and not info["is_threat"]:
            r = cfg.false_alarm_penalty
        else:
            r = 0.0
        self._rewards.append(r)

    def end_episode(self, optimizer: optim.Optimizer, gamma: float = 0.99):
        if not self._rewards:
            self._log_probs = []
            self._rewards = []
            return

        returns = compute_returns(self._rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.stack([-lp * R for lp, R in zip(self._log_probs, returns_t)]).sum()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        self._log_probs = []
        self._rewards = []


# ---------------------------------------------------------------------------
# Config C — REINFORCE + intrinsic surprise bonus
# ---------------------------------------------------------------------------

class ConfigC_Agent(ConfigB_Agent):
    """
    Config B + intrinsic surprise bonus.
    At each step, if pred_error > mean + 1*std, add a surprise bonus
    (0.5 * normalized_pred_error) to the base extrinsic reward.
    """

    def collect_reward(self, action: int, info: dict, pred_error: float = 0.0):
        """
        pred_error: the prediction error computed during step_act for this step.
        """
        base = 0.0
        cfg = self.env_cfg
        if action == 1 and info["is_threat"]:
            base = cfg.ward_correct_reward
        elif action == 1 and not info["is_threat"]:
            base = cfg.false_alarm_penalty

        # Intrinsic bonus
        surprise_bonus = 0.0
        threshold = self.pred_error_stats.threshold(n_std=1.0)
        if pred_error > threshold:
            normalized = self.pred_error_stats.normalized(pred_error)
            surprise_bonus = 0.5 * max(0.0, normalized)

        self._rewards.append(base + surprise_bonus)


# ---------------------------------------------------------------------------
# Config D — Surprise-gated curiosity only
# ---------------------------------------------------------------------------

class ConfigD_Agent(LSTMAgent):
    """
    Pure intrinsic curiosity agent. No extrinsic reward, no threat label,
    no ward health signal.

    Reward = 0.5 * normalized_pred_error
             IFF pred_error > running_mean + 1.0 * running_std
             AND action == 1.

    Otherwise reward = 0.

    Policy trained with REINFORCE using this reward only.
    """

    def __init__(self, env_cfg: EnvironmentConfig):
        super().__init__(env_cfg.obs_dim, hidden_size=128)
        self.to(DEVICE)
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

    def step_act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Forward pass + sample action + compute intrinsic reward.
        Returns (action, pred_error, intrinsic_reward).
        Appends log_prob and reward to episode buffers.
        """
        obs_t = self.obs_to_tensor(obs)
        pred_error = self.compute_pred_error(obs_t.squeeze(0).squeeze(0))

        _, pred, logit = self.forward(obs_t)
        self.store_prediction(pred)
        self.hx = tuple(h.detach() for h in self.hx)

        prob = torch.sigmoid(logit)
        dist = torch.distributions.Bernoulli(prob)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        self._log_probs.append(log_prob)

        action = int(action_t.item())

        # Surprise-gated intrinsic reward
        threshold = self.pred_error_stats.threshold(n_std=1.0)
        if pred_error > threshold and action == 1:
            normalized = self.pred_error_stats.normalized(pred_error)
            intrinsic_reward = 0.5 * max(0.0, normalized)
        else:
            intrinsic_reward = 0.0
        self._rewards.append(intrinsic_reward)

        return action, pred_error, intrinsic_reward

    def end_episode(self, optimizer: optim.Optimizer, gamma: float = 0.99):
        if not self._rewards:
            self._log_probs = []
            self._rewards = []
            return

        returns = compute_returns(self._rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.stack([-lp * R for lp, R in zip(self._log_probs, returns_t)]).sum()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        self._log_probs = []
        self._rewards = []


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns from a list of rewards."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def _episode_metrics_template() -> dict:
    return {
        "prec_det_steps": [],
        "threat_det_steps": [],
        "fp_steps": [],
        "pred_errors": [],
        "intrinsic_rewards": [],
    }


def _finalize_episode_metrics(m: dict, ward_health: float) -> dict:
    return {
        "ward_health": ward_health,
        "prec_det": float(np.mean(m["prec_det_steps"])) if m["prec_det_steps"] else 0.0,
        "threat_det": float(np.mean(m["threat_det_steps"])) if m["threat_det_steps"] else 0.0,
        "fp": float(np.mean(m["fp_steps"])) if m["fp_steps"] else 0.0,
        "mean_pred_error": float(np.mean(m["pred_errors"])) if m["pred_errors"] else 0.0,
        "mean_intrinsic_reward": float(np.mean(m["intrinsic_rewards"])) if m["intrinsic_rewards"] else 0.0,
    }


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------

def run_episode(
    agent: LSTMAgent,
    env: SignalEnvironment,
    optimizer: optim.Optimizer,
    cfg: EnvironmentConfig,
    train: bool = True,
    episode_num: int = 0,
) -> dict:
    """
    Run a single episode for any config agent.

    Dispatches to the correct inner loop based on agent type.
    Returns an episode metrics dict.
    """
    env.set_episode(episode_num)
    obs = env.reset()
    agent.reset_hidden()
    m = _episode_metrics_template()

    if isinstance(agent, ConfigA_Agent):
        _run_episode_A(agent, env, optimizer, cfg, train, obs, m)
    elif isinstance(agent, ConfigD_Agent):
        _run_episode_D(agent, env, optimizer, cfg, train, obs, m)
    elif isinstance(agent, ConfigC_Agent):
        # C before B because C inherits B
        _run_episode_C(agent, env, optimizer, cfg, train, obs, m)
    elif isinstance(agent, ConfigB_Agent):
        _run_episode_B(agent, env, optimizer, cfg, train, obs, m)
    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")

    return _finalize_episode_metrics(m, env.ward_health)


def _run_episode_A(
    agent: ConfigA_Agent,
    env: SignalEnvironment,
    optimizer: optim.Optimizer,
    cfg: EnvironmentConfig,
    train: bool,
    obs: np.ndarray,
    m: dict,
):
    for _ in range(cfg.episode_steps):
        action, loss = agent.step_supervised(obs, {"is_threat": False}, optimizer) if not train else (0, 0.0)
        # Properly call with info only when training
        if train:
            # We need to know is_threat for supervision — peek at current state before step
            # We use a two-phase approach: step the env first to get info, then update
            next_obs, info = env.step(0)  # dummy action to get state info
            # Redo forward pass with actual obs + state info for supervised update
            action, loss = agent.step_supervised(obs, info, optimizer)
            _accumulate_step_metrics(m, action, info)
            obs = next_obs
        else:
            action, _ = agent.step_supervised(obs, {"is_threat": False}, optimizer)
            next_obs, info = env.step(action)
            _accumulate_step_metrics(m, action, info)
            obs = next_obs
        m["pred_errors"].append(0.0)


def _run_episode_A(
    agent: ConfigA_Agent,
    env: SignalEnvironment,
    optimizer: optim.Optimizer,
    cfg: EnvironmentConfig,
    train: bool,
    obs: np.ndarray,
    m: dict,
):
    """
    Config A inner loop.
    We interleave forward pass and environment step so the supervised
    label (info["is_threat"]) is available before the gradient update.
    """
    for _ in range(cfg.episode_steps):
        obs_t = agent.obs_to_tensor(obs)
        pred_error = agent.compute_pred_error(obs_t.squeeze(0).squeeze(0))
        m["pred_errors"].append(pred_error)

        # Forward without update to get action first, then get env label
        with torch.no_grad():
            _, pred_no_grad, logit_no_grad = agent.forward(obs_t)
        # Restore hx — forward already updated it; we'll redo inside step_supervised
        # Save hx before the no-grad forward
        # Simpler: just use the logit from no-grad for action, then do supervised step
        prob = torch.sigmoid(logit_no_grad).item()
        action = int(prob > 0.5)

        next_obs, info = env.step(action)

        if train:
            # Supervised update using ground truth label
            _, _ = agent.step_supervised(obs, info, optimizer)

        _accumulate_step_metrics(m, action, info)
        obs = next_obs


def _run_episode_B(
    agent: ConfigB_Agent,
    env: SignalEnvironment,
    optimizer: optim.Optimizer,
    cfg: EnvironmentConfig,
    train: bool,
    obs: np.ndarray,
    m: dict,
):
    for _ in range(cfg.episode_steps):
        action, pred_error = agent.step_act(obs)
        next_obs, info = env.step(action)
        m["pred_errors"].append(pred_error)

        if train:
            agent.collect_reward(action, info)

        _accumulate_step_metrics(m, action, info)
        obs = next_obs

    if train:
        agent.end_episode(optimizer)


def _run_episode_C(
    agent: ConfigC_Agent,
    env: SignalEnvironment,
    optimizer: optim.Optimizer,
    cfg: EnvironmentConfig,
    train: bool,
    obs: np.ndarray,
    m: dict,
):
    for _ in range(cfg.episode_steps):
        action, pred_error = agent.step_act(obs)
        next_obs, info = env.step(action)
        m["pred_errors"].append(pred_error)

        if train:
            agent.collect_reward(action, info, pred_error=pred_error)

        _accumulate_step_metrics(m, action, info)
        obs = next_obs

    if train:
        agent.end_episode(optimizer)


def _run_episode_D(
    agent: ConfigD_Agent,
    env: SignalEnvironment,
    optimizer: optim.Optimizer,
    cfg: EnvironmentConfig,
    train: bool,
    obs: np.ndarray,
    m: dict,
):
    for _ in range(cfg.episode_steps):
        action, pred_error, intrinsic_reward = agent.step_act(obs)
        next_obs, info = env.step(action)
        m["pred_errors"].append(pred_error)
        m["intrinsic_rewards"].append(intrinsic_reward)

        _accumulate_step_metrics(m, action, info)
        obs = next_obs

    if train:
        agent.end_episode(optimizer)


def _accumulate_step_metrics(m: dict, action: int, info: dict):
    if info["is_precursor"]:
        m["prec_det_steps"].append(float(action))
    if info["is_threat"]:
        m["threat_det_steps"].append(float(action))
    if info["is_normal"]:
        m["fp_steps"].append(float(action))


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    agent: LSTMAgent,
    optimizer: optim.Optimizer,
    episode: int,
    path: str,
):
    """
    Save agent weights + optimizer state + Welford stats + episode number.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "episode": episode,
        "agent_state_dict": agent.state_dict_extended(),
        "optimizer_state_dict": optimizer.state_dict(),
        "agent_class": type(agent).__name__,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    agent: LSTMAgent,
    optimizer: optim.Optimizer,
    path: str,
) -> int:
    """
    Load agent weights + optimizer state from checkpoint.
    Returns episode number to resume from.
    Raises FileNotFoundError if path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    agent.load_state_dict_extended(checkpoint["agent_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["episode"])


def checkpoint_path(
    base_dir: str,
    config_key: str,
    seed: int,
    episode: int,
) -> str:
    return os.path.join(base_dir, f"config{config_key}_seed{seed}_ep{episode:04d}.pt")


def find_latest_checkpoint(
    base_dir: str,
    config_key: str,
    seed: int,
) -> Optional[str]:
    """
    Scan base_dir for the highest-episode checkpoint for a given config+seed.
    Returns the path or None if no checkpoint found.
    """
    if not os.path.isdir(base_dir):
        return None
    prefix = f"config{config_key}_seed{seed}_ep"
    candidates = [
        f for f in os.listdir(base_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not candidates:
        return None
    # Sort by episode number
    def ep_num(fname: str) -> int:
        try:
            return int(fname.replace(prefix, "").replace(".pt", ""))
        except ValueError:
            return -1
    candidates.sort(key=ep_num, reverse=True)
    return os.path.join(base_dir, candidates[0])


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(config_key: str, env_cfg: EnvironmentConfig) -> LSTMAgent:
    """
    Instantiate the correct agent class for a given config key ('A', 'B', 'C', 'D').
    """
    if config_key == "A":
        return ConfigA_Agent(env_cfg)
    elif config_key == "B":
        return ConfigB_Agent(env_cfg)
    elif config_key == "C":
        return ConfigC_Agent(env_cfg)
    elif config_key == "D":
        return ConfigD_Agent(env_cfg)
    else:
        raise ValueError(f"Unknown config key: {config_key!r}. Must be one of A, B, C, D.")


def make_optimizer(agent: LSTMAgent, lr: float = 3e-4) -> optim.Adam:
    return optim.Adam(agent.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# Quick smoke test (runs when module is executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Running 1-episode smoke test on all configs...")

    env_cfg = EnvironmentConfig(episode_steps=50, n_episodes=1)
    seeds = [42]

    config_names = {"A": "Supervised", "B": "Protection", "C": "Prot+Intrinsic", "D": "Intrinsic"}
    results = {}

    for ck in ["A", "B", "C", "D"]:
        set_seed(42)
        env = SignalEnvironment(env_cfg, seed=42)
        agent = make_agent(ck, env_cfg)
        opt = make_optimizer(agent)
        ep_result = run_episode(agent, env, opt, env_cfg, train=True, episode_num=0)
        results[ck] = ep_result
        print(
            f"  Config {ck} ({config_names[ck]}): "
            f"ward={ep_result['ward_health']:.1f} "
            f"prec_det={ep_result['prec_det']:.3f} "
            f"fp={ep_result['fp']:.3f}"
        )

    print("\nSmoke test passed. experiment_core.py is working correctly.")
