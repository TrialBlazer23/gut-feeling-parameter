"""
GFP Round 2 shared environment, agent, and utilities.
Key change from Round 1: WINDOW default updated to 25 (from Exp 3 result).
Imported by exp6, exp7, exp8.
"""

import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────

def make_episode(n_steps=400, window=25, prec_shift=0.08, threat_shift=0.35,
                 adversarial=False, decoy_mag=0.22, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    sig = rng.normal(0, 1.0, n_steps).astype(np.float32)
    lbl = np.zeros(n_steps, dtype=np.int64)
    t = 0
    while t < n_steps:
        if adversarial and rng.random() < 0.03:
            dur = rng.randint(4, 10)
            sig[t:min(t + dur, n_steps)] += decoy_mag
            t += dur
            continue
        if rng.random() < 0.04 and t + 15 < n_steps:
            ps = prec_shift * 0.5 if adversarial else prec_shift
            sig[t:t + 15] += ps
            lbl[t:t + 15] = 1
            t += 15
            if rng.random() < 0.70:
                gap = rng.randint(5, 18)
                s2 = t + gap
                e2 = min(s2 + 8, n_steps)
                if s2 < n_steps:
                    sig[s2:e2] += threat_shift
                    lbl[s2:e2] = 2
                    t = e2
        else:
            t += 1
    return sig, lbl


# ─── AGENT ────────────────────────────────────────────────────────────────────

class FastAgent(nn.Module):
    """
    MLP over flattened window.
    Supports Config D (surprise-gated intrinsic) and Config B (protection).
    pool_ctx_dim: 0 = no pool, 2 = (mean_surp, prec_frac) from shared pool.
    """

    def __init__(self, window: int = 25, hidden: int = 24,
                 std_mult: float = 1.0, pool_ctx_dim: int = 0):
        super().__init__()
        self.window = window
        self.std_mult = std_mult
        self.pool_ctx_dim = pool_ctx_dim
        in_dim = window + pool_ctx_dim
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pred_head = nn.Linear(hidden, 1)
        self.policy_head = nn.Linear(hidden, 2)
        # Welford running statistics
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def _welford(self, x):
        self._n += 1
        d = x - self._mean
        self._mean += d / self._n
        self._M2 += d * (x - self._mean)

    @property
    def err_std(self):
        return math.sqrt(max(self._M2 / max(self._n - 1, 1), 1e-8))

    def forward(self, obs_flat, pool_ctx=None):
        if self.pool_ctx_dim > 0:
            if pool_ctx is not None:
                ctx_t = torch.FloatTensor([[pool_ctx[0], pool_ctx[1]]])
            else:
                ctx_t = torch.zeros(1, self.pool_ctx_dim)
            inp = torch.cat([obs_flat, ctx_t], dim=-1)
        else:
            inp = obs_flat
        h = self.enc(inp)
        return h, self.pred_head(h), self.policy_head(h)

    def intrinsic_D(self, pred_val, actual, action):
        err = abs(pred_val - actual)
        self._welford(err)
        norm = (err - self._mean) / (self.err_std + 1e-8)
        gate = err > (self._mean + self.std_mult * self.err_std)
        if gate and action == 1:
            return 0.5 * max(norm, 0.0), err, True
        return 0.0, err, False

    def intrinsic_B(self, action, true_label):
        if true_label == 2 and action == 1:
            return 1.0
        if action == 1 and true_label == 0:
            return -0.3
        return 0.0

    def welford_n(self):
        """Return current Welford sample count (proxy for convergence)."""
        return self._n


# ─── SHARED SURPRISE POOL (Exp 8) ─────────────────────────────────────────────

class Pool:
    def __init__(self, cap=500, dim=24):
        self.cap = cap
        self.dim = dim
        self.K = np.zeros((cap, dim), np.float32)
        self.S = np.zeros(cap, np.float32)
        self.L = np.zeros(cap, np.int64)
        self.ptr = 0
        self.n = 0

    def write(self, h_np, surp, lbl):
        self.K[self.ptr] = h_np
        self.S[self.ptr] = surp
        self.L[self.ptr] = lbl
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def query(self, h_np, k=5):
        if self.n == 0:
            return 0.0, 0.0
        q = h_np / (np.linalg.norm(h_np) + 1e-8)
        Kn = self.K[:self.n]
        norms = np.linalg.norm(Kn, axis=1, keepdims=True) + 1e-8
        sims = (Kn / norms) @ q
        top = np.argsort(sims)[-(min(k, self.n)):]
        return float(self.S[top].mean()), float((self.L[top] == 1).mean())

    def size(self):
        return self.n


# ─── EPISODE RUNNER ───────────────────────────────────────────────────────────

def run_ep(agent, sig, lbl, optimizer, window=25, mode="D",
           pool=None, writes_to_pool=False, train=True):
    """
    Run one episode. Returns per-episode detection stats.
    pool: Pool instance (used in Exp 8)
    writes_to_pool: if True, agent writes surprise events to pool
    """
    log_probs, rewards = [], []
    prec_det = prec_tot = thr_det = thr_tot = surp_n = 0

    for t in range(window, len(sig)):
        obs = torch.FloatTensor(sig[t - window:t]).unsqueeze(0)  # (1, window)

        ctx = None
        if agent.pool_ctx_dim > 0 and pool is not None and pool.n > 0:
            with torch.no_grad():
                h_q, _, _ = agent(obs, pool_ctx=None)
            ctx = pool.query(h_q.numpy().squeeze())

        h, pred, logits = agent(obs, pool_ctx=ctx)
        dist = torch.distributions.Categorical(F.softmax(logits, -1))
        a = dist.sample()
        lp = dist.log_prob(a)
        a_val = a.item()

        actual = float(sig[t]) if t < len(sig) else 0.0

        if mode == "D":
            r, err, surprised = agent.intrinsic_D(pred.item(), actual, a_val)
            if surprised and writes_to_pool and pool is not None:
                pool.write(h.detach().numpy().squeeze(), err, int(lbl[t - 1]))
        else:
            r = agent.intrinsic_B(a_val, int(lbl[t - 1]))
            surprised = False

        cur_lbl = int(lbl[t - 1])
        if cur_lbl == 1:
            prec_tot += 1
            prec_det += (a_val == 1)
        if cur_lbl == 2:
            thr_tot += 1
            thr_det += (a_val == 1)
        if surprised:
            surp_n += 1

        log_probs.append(lp)
        rewards.append(r)

    if train and any(abs(r) > 1e-8 for r in rewards):
        G = 0.0
        rets = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            rets.insert(0, G)
        rets_t = torch.FloatTensor(rets)
        if rets_t.std() > 1e-8:
            rets_t = (rets_t - rets_t.mean()) / (rets_t.std() + 1e-8)
        loss = -sum(lp * r for lp, r in zip(log_probs, rets_t))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

    return {
        "prec_det": prec_det / max(prec_tot, 1),
        "thr_det": thr_det / max(thr_tot, 1),
        "surp_rate": surp_n / max(len(rewards), 1),
    }


# ─── UTILITIES ─────────────────────────────────────────────────────────────────

def last_n(ep_list, n=8, key="prec_det"):
    return float(np.mean([e[key] for e in ep_list[-n:]])) if ep_list else 0.0


def smooth(arr, w=4):
    if len(arr) < w:
        return list(arr)
    return list(np.convolve(arr, np.ones(w) / w, mode="valid"))


# ─── DEFAULTS ──────────────────────────────────────────────────────────────────
SEEDS = [42, 43, 44]
WINDOW = 25       # Updated from 50 → 25 based on Exp 3
N_STEPS = 400
