"""
EXP 6 — Welford Stabilization Lag
===================================
Question: Does threshold sensitivity (std_mult effect) emerge only AFTER
Welford running statistics have stabilized?

Design:
  - Sweep std_mult in [0.25, 0.5, 1.0, 1.5, 2.0]
  - Two episode-count regimes:
      SHORT: 30 episodes (Welford is still warming up)
      LONG:  150 episodes (Welford has converged)
  - Config D only (the parameter we care about)
  - window=25, 3 seeds, N_STEPS=400

Hypothesis: The std_mult effect is suppressed in SHORT regime because Welford
hasn't seen enough samples to distinguish multiplier values. In LONG regime,
the threshold becomes meaningfully different across multipliers, and
performance should diverge.

Output:
  - gfp_exp_results/exp6_summary.json
  - gfp_exp_results/exp6_welford_lag.png
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
    FastAgent, make_episode, run_ep,
    set_seed, last_n, smooth,
    SEEDS, WINDOW, N_STEPS,
)

OUT_DIR = "/home/user/workspace/gfp_exp_results"
os.makedirs(OUT_DIR, exist_ok=True)

STD_MULTS = [0.25, 0.5, 1.0, 1.5, 2.0]
N_SHORT = 30
N_LONG = 150
HIDDEN = 24
LR = 3e-3


def run_condition(std_mult, n_ep, seed):
    set_seed(seed)
    rng = np.random.RandomState(seed)
    agent = FastAgent(window=WINDOW, hidden=HIDDEN, std_mult=std_mult)
    opt = optim.Adam(agent.parameters(), lr=LR)
    history = []
    for ep in range(n_ep):
        sig, lbl = make_episode(n_steps=N_STEPS, window=WINDOW, rng=rng)
        res = run_ep(agent, sig, lbl, opt, window=WINDOW, mode="D")
        history.append(res)
    return last_n(history, n=8, key="prec_det"), history


print("EXP 6: Welford Stabilization Lag")
print(f"  std_mults={STD_MULTS}")
print(f"  SHORT={N_SHORT} ep | LONG={N_LONG} ep | window={WINDOW} | seeds={SEEDS}")

results = {"short": {}, "long": {}}
curves_short = {}
curves_long = {}

for regime_name, n_ep in [("short", N_SHORT), ("long", N_LONG)]:
    print(f"\n  --- {regime_name.upper()} REGIME ({n_ep} ep) ---")
    curves = {}
    for mult in STD_MULTS:
        seed_scores = []
        seed_curves = []
        for seed in SEEDS:
            score, history = run_condition(mult, n_ep, seed)
            seed_scores.append(score)
            seed_curves.append([e["prec_det"] for e in history])
        mean_s = float(np.mean(seed_scores))
        std_s = float(np.std(seed_scores))
        results[regime_name][str(mult)] = {
            "mean": mean_s,
            "std": std_s,
            "per_seed": seed_scores,
        }
        # Average curve across seeds (truncate to min length for safety)
        min_len = min(len(c) for c in seed_curves)
        arr = np.array([c[:min_len] for c in seed_curves])
        curves[mult] = arr.mean(axis=0).tolist()
        print(f"    std_mult={mult:.2f} → prec_det={mean_s:.4f} ± {std_s:.4f}")
    if regime_name == "short":
        curves_short = curves
    else:
        curves_long = curves

# ─── ANALYSIS ─────────────────────────────────────────────────────────────────
short_means = [results["short"][str(m)]["mean"] for m in STD_MULTS]
long_means  = [results["long"][str(m)]["mean"]  for m in STD_MULTS]

short_range = max(short_means) - min(short_means)
long_range  = max(long_means)  - min(long_means)

results["analysis"] = {
    "short_range": short_range,
    "long_range":  long_range,
    "range_increase": long_range - short_range,
    "hypothesis_supported": long_range > short_range * 1.5,
    "short_best_mult": STD_MULTS[int(np.argmax(short_means))],
    "long_best_mult":  STD_MULTS[int(np.argmax(long_means))],
}

print(f"\n  ANALYSIS:")
print(f"    Short regime score range: {short_range:.4f}")
print(f"    Long  regime score range: {long_range:.4f}")
print(f"    Range increase: {long_range - short_range:.4f}")
print(f"    Hypothesis supported: {results['analysis']['hypothesis_supported']}")

# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(STD_MULTS)))

# Panel A: Short regime bar chart
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor("#1a1a1a")
bars = ax_a.bar([str(m) for m in STD_MULTS], short_means, color=colors, alpha=0.85, width=0.6)
errs_s = [results["short"][str(m)]["std"] for m in STD_MULTS]
ax_a.errorbar([str(m) for m in STD_MULTS], short_means, yerr=errs_s,
              fmt="none", color="white", capsize=4, linewidth=1.2)
ax_a.set_title(f"Short Regime ({N_SHORT} ep) — std_mult sweep", color="white", fontsize=10, pad=8)
ax_a.set_xlabel("std_mult", color="#aaaaaa", fontsize=9)
ax_a.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_a.tick_params(colors="white")
for spine in ax_a.spines.values():
    spine.set_edgecolor("#444444")
for i, (bar, val) in enumerate(zip(bars, short_means)):
    ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
              f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=8)

# Panel B: Long regime bar chart
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor("#1a1a1a")
bars2 = ax_b.bar([str(m) for m in STD_MULTS], long_means, color=colors, alpha=0.85, width=0.6)
errs_l = [results["long"][str(m)]["std"] for m in STD_MULTS]
ax_b.errorbar([str(m) for m in STD_MULTS], long_means, yerr=errs_l,
              fmt="none", color="white", capsize=4, linewidth=1.2)
ax_b.set_title(f"Long Regime ({N_LONG} ep) — std_mult sweep", color="white", fontsize=10, pad=8)
ax_b.set_xlabel("std_mult", color="#aaaaaa", fontsize=9)
ax_b.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_b.tick_params(colors="white")
for spine in ax_b.spines.values():
    spine.set_edgecolor("#444444")
for i, (bar, val) in enumerate(zip(bars2, long_means)):
    ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
              f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=8)

# Panel C: Short regime learning curves
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor("#1a1a1a")
for i, mult in enumerate(STD_MULTS):
    c = smooth(curves_short[mult], w=4)
    ax_c.plot(c, color=colors[i], alpha=0.85, linewidth=1.4, label=f"×{mult}")
ax_c.set_title(f"Learning Curves — Short ({N_SHORT} ep)", color="white", fontsize=10, pad=8)
ax_c.set_xlabel("Episode (smoothed)", color="#aaaaaa", fontsize=9)
ax_c.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_c.tick_params(colors="white")
ax_c.legend(fontsize=7, facecolor="#222222", labelcolor="white", framealpha=0.7,
            title="std_mult", title_fontsize=7)
for spine in ax_c.spines.values():
    spine.set_edgecolor("#444444")

# Panel D: Long regime learning curves
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor("#1a1a1a")
for i, mult in enumerate(STD_MULTS):
    c = smooth(curves_long[mult], w=8)
    ax_d.plot(c, color=colors[i], alpha=0.85, linewidth=1.4, label=f"×{mult}")
ax_d.set_title(f"Learning Curves — Long ({N_LONG} ep)", color="white", fontsize=10, pad=8)
ax_d.set_xlabel("Episode (smoothed)", color="#aaaaaa", fontsize=9)
ax_d.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_d.tick_params(colors="white")
ax_d.legend(fontsize=7, facecolor="#222222", labelcolor="white", framealpha=0.7,
            title="std_mult", title_fontsize=7)
for spine in ax_d.spines.values():
    spine.set_edgecolor("#444444")

# Suptitle
supported_str = "SUPPORTED" if results["analysis"]["hypothesis_supported"] else "NOT SUPPORTED"
fig.suptitle(
    f"EXP 6 — Welford Stabilization Lag  |  window=25  |  3 seeds\n"
    f"Score range SHORT={short_range:.4f} → LONG={long_range:.4f}  |  Hypothesis: {supported_str}",
    color="white", fontsize=11, y=0.98
)

fig_path = os.path.join(OUT_DIR, "exp6_welford_lag.png")
plt.savefig(fig_path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"\n  Figure saved: {fig_path}")

# ─── SAVE JSON ────────────────────────────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "exp6_summary.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  JSON saved:  {json_path}")
print("\nEXP 6 COMPLETE.")
