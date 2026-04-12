"""
EXP 8 — Inter-Episode Contagion (Redesign)
============================================
Question: Can a second agent bootstrap faster by starting with a pre-populated
surprise pool from a first agent that already trained?

Design:
  Phase 1 (Pre-training):
    Agent A1 trains for PRE_EP episodes alone, writing all surprise events
    to a shared Pool. Pool is then frozen.

  Phase 2 (Test):
    Agent A2: trains from scratch for TEST_EP episodes WITH read access to A1's pool.
    Agent A3: trains from scratch for TEST_EP episodes WITHOUT pool access (control).

  Measurement:
    - Per-episode precursor detection for A2 and A3 across TEST_EP episodes
    - Convergence speed: episode at which each first reaches 0.4 prec_det
    - Final performance: last_n(8) of TEST_EP
    - Pool size at start of Phase 2 (proxy for how much A1 contributed)

  Hypothesis: A2 converges faster than A3 because A1's pool encodes precursor-
  associated hidden states that A2 can query to get useful context before its
  own Welford statistics have stabilized.

  window=25, hidden=24, 3 seeds, prec_shift=0.08 (standard).

Output:
  - gfp_exp_results/exp8_summary.json
  - gfp_exp_results/exp8_contagion_redesign.png
"""

import sys, json, os
sys.path.insert(0, "/home/user/workspace")

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gfp_shared_r2 import (
    FastAgent, Pool, make_episode, run_ep,
    set_seed, last_n, smooth,
    SEEDS, WINDOW, N_STEPS,
)

OUT_DIR = "/home/user/workspace/gfp_exp_results"
os.makedirs(OUT_DIR, exist_ok=True)

PRE_EP = 50     # A1 pre-training episodes to populate pool
TEST_EP = 60    # Episodes for A2 (with pool) and A3 (control) to train
HIDDEN = 24
LR = 3e-3
POOL_CAP = 500
CONVERGENCE_THRESHOLD = 0.40   # prec_det value to define "converged"


def first_convergence_episode(ep_scores, threshold=CONVERGENCE_THRESHOLD, window=5):
    """Return first episode index where rolling mean of window episodes >= threshold."""
    for i in range(window, len(ep_scores)):
        if np.mean(ep_scores[i - window:i]) >= threshold:
            return i
    return None   # never converged


def run_seed(seed):
    rng = np.random.RandomState(seed)

    # ── Phase 1: A1 pre-trains and populates pool ──
    set_seed(seed)
    pool = Pool(cap=POOL_CAP, dim=HIDDEN)
    a1 = FastAgent(window=WINDOW, hidden=HIDDEN, std_mult=1.0, pool_ctx_dim=0)
    opt1 = optim.Adam(a1.parameters(), lr=LR)
    for ep in range(PRE_EP):
        sig, lbl = make_episode(n_steps=N_STEPS, window=WINDOW, rng=rng)
        run_ep(a1, sig, lbl, opt1, window=WINDOW, mode="D",
               pool=pool, writes_to_pool=True, train=True)

    pool_size_at_start = pool.size()

    # ── Phase 2: A2 trains with frozen pool ──
    set_seed(seed + 1000)
    rng2 = np.random.RandomState(seed + 1000)
    a2 = FastAgent(window=WINDOW, hidden=HIDDEN, std_mult=1.0, pool_ctx_dim=2)
    opt2 = optim.Adam(a2.parameters(), lr=LR)
    a2_history = []
    for ep in range(TEST_EP):
        sig, lbl = make_episode(n_steps=N_STEPS, window=WINDOW, rng=rng2)
        # Pass pool for reading but A2 does NOT write to it (pool is frozen)
        res = run_ep(a2, sig, lbl, opt2, window=WINDOW, mode="D",
                     pool=pool, writes_to_pool=False, train=True)
        a2_history.append(res)

    # ── Phase 2: A3 trains without pool (control) ──
    set_seed(seed + 2000)
    rng3 = np.random.RandomState(seed + 2000)
    a3 = FastAgent(window=WINDOW, hidden=HIDDEN, std_mult=1.0, pool_ctx_dim=0)
    opt3 = optim.Adam(a3.parameters(), lr=LR)
    a3_history = []
    for ep in range(TEST_EP):
        sig, lbl = make_episode(n_steps=N_STEPS, window=WINDOW, rng=rng3)
        res = run_ep(a3, sig, lbl, opt3, window=WINDOW, mode="D",
                     pool=None, writes_to_pool=False, train=True)
        a3_history.append(res)

    a2_scores = [e["prec_det"] for e in a2_history]
    a3_scores = [e["prec_det"] for e in a3_history]

    return {
        "pool_size": pool_size_at_start,
        "a2_final": last_n(a2_history, n=8, key="prec_det"),
        "a3_final": last_n(a3_history, n=8, key="prec_det"),
        "a2_convergence_ep": first_convergence_episode(a2_scores),
        "a3_convergence_ep": first_convergence_episode(a3_scores),
        "a2_scores": a2_scores,
        "a3_scores": a3_scores,
    }


print("EXP 8: Inter-Episode Contagion (Redesign)")
print(f"  PRE_EP={PRE_EP} (A1 pool fill) | TEST_EP={TEST_EP} | window={WINDOW} | seeds={SEEDS}")

all_results = {}
for seed in SEEDS:
    print(f"\n  --- SEED {seed} ---")
    r = run_seed(seed)
    all_results[str(seed)] = r
    lift = r["a2_final"] - r["a3_final"]
    conv_a2 = r["a2_convergence_ep"] or "never"
    conv_a3 = r["a3_convergence_ep"] or "never"
    print(f"    Pool size at start: {r['pool_size']}")
    print(f"    A2 (pool) final:   {r['a2_final']:.4f}  | converged ep: {conv_a2}")
    print(f"    A3 (control) final:{r['a3_final']:.4f}  | converged ep: {conv_a3}")
    print(f"    Contagion lift:    {lift:+.4f}")

# Aggregate
a2_finals = [all_results[str(s)]["a2_final"] for s in SEEDS]
a3_finals = [all_results[str(s)]["a3_final"] for s in SEEDS]
lifts = [a2 - a3 for a2, a3 in zip(a2_finals, a3_finals)]
pool_sizes = [all_results[str(s)]["pool_size"] for s in SEEDS]

# Convergence speed: average episode of convergence (None → TEST_EP as upper bound)
def conv_val(v):
    return v if v is not None else TEST_EP

a2_convs = [conv_val(all_results[str(s)]["a2_convergence_ep"]) for s in SEEDS]
a3_convs = [conv_val(all_results[str(s)]["a3_convergence_ep"]) for s in SEEDS]
conv_speedup = [a3 - a2 for a2, a3 in zip(a2_convs, a3_convs)]   # positive = A2 faster

results = {
    "pre_ep": PRE_EP,
    "test_ep": TEST_EP,
    "per_seed": {str(s): {k: v for k, v in all_results[str(s)].items()
                          if k not in ("a2_scores", "a3_scores")}
                 for s in SEEDS},
    "aggregate": {
        "a2_final_mean": float(np.mean(a2_finals)),
        "a2_final_std":  float(np.std(a2_finals)),
        "a3_final_mean": float(np.mean(a3_finals)),
        "a3_final_std":  float(np.std(a3_finals)),
        "lift_mean": float(np.mean(lifts)),
        "lift_std":  float(np.std(lifts)),
        "lift_per_seed": lifts,
        "pool_size_mean": float(np.mean(pool_sizes)),
        "conv_speedup_mean": float(np.mean(conv_speedup)),   # ep A2 converges earlier
        "conv_speedup_per_seed": conv_speedup,
    },
    "analysis": {
        "hypothesis_supported": float(np.mean(lifts)) > 0.01,
        "a2_faster_convergence": float(np.mean(conv_speedup)) > 2,
        "pool_useful": float(np.mean(lifts)) > 0,
    }
}

print(f"\n  AGGREGATE:")
print(f"    A2 (pool) final:    {results['aggregate']['a2_final_mean']:.4f} ± {results['aggregate']['a2_final_std']:.4f}")
print(f"    A3 (ctrl) final:    {results['aggregate']['a3_final_mean']:.4f} ± {results['aggregate']['a3_final_std']:.4f}")
print(f"    Mean lift:          {results['aggregate']['lift_mean']:+.4f}")
print(f"    Conv speedup (ep):  {results['aggregate']['conv_speedup_mean']:+.2f}")
print(f"    Hypothesis supported: {results['analysis']['hypothesis_supported']}")

# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

col_a2 = "#00bfff"
col_a3 = "#ff9f43"
seed_colors = ["#bd93f9", "#ff79c6", "#50fa7b"]

# Panel A: Per-seed learning curves A2 vs A3
ax_a = fig.add_subplot(gs[0, :])   # full-width top panel
ax_a.set_facecolor("#1a1a1a")
for i, seed in enumerate(SEEDS):
    a2s = smooth(all_results[str(seed)]["a2_scores"], w=5)
    a3s = smooth(all_results[str(seed)]["a3_scores"], w=5)
    ax_a.plot(a2s, color=col_a2, alpha=0.5 + 0.15 * i, linewidth=1.2,
              label=f"A2 pool s{seed}" if i == 0 else f"_A2 s{seed}")
    ax_a.plot(a3s, color=col_a3, alpha=0.5 + 0.15 * i, linewidth=1.2, linestyle="--",
              label=f"A3 ctrl s{seed}" if i == 0 else f"_A3 s{seed}")
ax_a.axhline(CONVERGENCE_THRESHOLD, color="#888888", linewidth=0.8, linestyle=":",
             label=f"convergence threshold ({CONVERGENCE_THRESHOLD})")
ax_a.set_title("A2 (pre-populated pool) vs A3 (control) — Per-Seed Learning Curves",
               color="white", fontsize=10, pad=8)
ax_a.set_xlabel("Episode (smoothed)", color="#aaaaaa", fontsize=9)
ax_a.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_a.tick_params(colors="white")
ax_a.legend(fontsize=7, facecolor="#222222", labelcolor="white", framealpha=0.7, ncol=3)
for spine in ax_a.spines.values():
    spine.set_edgecolor("#444444")

# Panel B: Final performance bar chart
ax_b = fig.add_subplot(gs[1, 0])
ax_b.set_facecolor("#1a1a1a")
xpos = np.arange(len(SEEDS))
w = 0.35
ax_b.bar(xpos - w / 2, a2_finals, w, color=col_a2, alpha=0.85, label="A2 (pool)")
ax_b.bar(xpos + w / 2, a3_finals, w, color=col_a3, alpha=0.85, label="A3 (ctrl)")
ax_b.set_xticks(xpos)
ax_b.set_xticklabels([f"seed {s}" for s in SEEDS])
ax_b.set_title("Final Performance (last 8 ep)", color="white", fontsize=10, pad=8)
ax_b.set_xlabel("Seed", color="#aaaaaa", fontsize=9)
ax_b.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_b.tick_params(colors="white")
ax_b.legend(fontsize=8, facecolor="#222222", labelcolor="white", framealpha=0.7)
for spine in ax_b.spines.values():
    spine.set_edgecolor("#444444")

# Panel C: Convergence speed and lift
ax_c = fig.add_subplot(gs[1, 1])
ax_c.set_facecolor("#1a1a1a")
lift_colors = ["#50fa7b" if l > 0 else "#ff5555" for l in lifts]
ax_c.bar([f"seed {s}" for s in SEEDS], lifts, color=lift_colors, alpha=0.85, width=0.5)
ax_c.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
ax_c.set_title("Contagion Lift per Seed (A2 − A3)", color="white", fontsize=10, pad=8)
ax_c.set_xlabel("Seed", color="#aaaaaa", fontsize=9)
ax_c.set_ylabel("A2 − A3 (prec_det)", color="#aaaaaa", fontsize=9)
ax_c.tick_params(colors="white")
for spine in ax_c.spines.values():
    spine.set_edgecolor("#444444")
for i, (val, s) in enumerate(zip(lifts, SEEDS)):
    ax_c.text(i, val + (0.001 if val >= 0 else -0.004),
              f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
              color="white", fontsize=9)

supported_str = "SUPPORTED" if results["analysis"]["hypothesis_supported"] else "NOT SUPPORTED"
lift_mean = results["aggregate"]["lift_mean"]
pool_mean = results["aggregate"]["pool_size_mean"]
conv_spd = results["aggregate"]["conv_speedup_mean"]

fig.suptitle(
    f"EXP 8 — Inter-Episode Contagion Redesign  |  window=25  |  3 seeds\n"
    f"A1 pre-trains {PRE_EP} ep  →  pool size ≈{pool_mean:.0f}  |  "
    f"Mean lift={lift_mean:+.4f}  |  Conv speedup={conv_spd:+.1f} ep  |  "
    f"Hypothesis: {supported_str}",
    color="white", fontsize=10, y=0.98
)

fig_path = os.path.join(OUT_DIR, "exp8_contagion_redesign.png")
plt.savefig(fig_path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"\n  Figure saved: {fig_path}")

json_path = os.path.join(OUT_DIR, "exp8_summary.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  JSON saved:  {json_path}")
print("\nEXP 8 COMPLETE.")
