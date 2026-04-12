"""
EXP 7 — Signal Detection Floor
================================
Question: Does Config D still outperform Config B below mean_shift=0.02?
Where is the true detection floor?

Design:
  - mean_shift values: [0.005, 0.01, 0.015, 0.02]
    (all weaker than the already-subtle Exp 2 minimum of 0.02)
  - Config D vs Config B head-to-head at each shift level
  - window=25 (corrected default), hidden=24, 40 episodes, 3 seeds
  - threat_shift kept at 0.35 (unchanged) — only precursor signal varies

Hypothesis: Config D continues to detect precursors even at sub-0.02 shift
levels where Config B's detection collapses toward chance. The D-vs-B gap
widens or stays constant as signal weakens — it does not reverse.

Output:
  - gfp_exp_results/exp7_summary.json
  - gfp_exp_results/exp7_detection_floor.png
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

SHIFTS = [0.005, 0.01, 0.015, 0.02]
N_EP = 40
HIDDEN = 24
LR = 3e-3
STD_MULT = 1.0


def run_condition(shift, mode, seed):
    set_seed(seed)
    rng = np.random.RandomState(seed)
    agent = FastAgent(window=WINDOW, hidden=HIDDEN, std_mult=STD_MULT)
    opt = optim.Adam(agent.parameters(), lr=LR)
    history = []
    for ep in range(N_EP):
        sig, lbl = make_episode(n_steps=N_STEPS, window=WINDOW,
                                prec_shift=shift, threat_shift=0.35, rng=rng)
        res = run_ep(agent, sig, lbl, opt, window=WINDOW, mode=mode)
        history.append(res)
    return last_n(history, n=8, key="prec_det"), history


print("EXP 7: Signal Detection Floor")
print(f"  shifts={SHIFTS}")
print(f"  configs=[D, B] | {N_EP} ep | window={WINDOW} | seeds={SEEDS}")

results = {"D": {}, "B": {}, "gaps": {}}
curves_D = {}
curves_B = {}

for mode in ["D", "B"]:
    print(f"\n  --- CONFIG {mode} ---")
    for shift in SHIFTS:
        seed_scores = []
        seed_curves = []
        for seed in SEEDS:
            score, history = run_condition(shift, mode, seed)
            seed_scores.append(score)
            seed_curves.append([e["prec_det"] for e in history])
        mean_s = float(np.mean(seed_scores))
        std_s = float(np.std(seed_scores))
        results[mode][str(shift)] = {
            "mean": mean_s,
            "std": std_s,
            "per_seed": seed_scores,
        }
        min_len = min(len(c) for c in seed_curves)
        arr = np.array([c[:min_len] for c in seed_curves])
        if mode == "D":
            curves_D[shift] = arr.mean(axis=0).tolist()
        else:
            curves_B[shift] = arr.mean(axis=0).tolist()
        print(f"    shift={shift:.3f} → prec_det={mean_s:.4f} ± {std_s:.4f}")

# Compute D-vs-B gap at each shift
print("\n  --- D vs B GAPS ---")
for shift in SHIFTS:
    d_mean = results["D"][str(shift)]["mean"]
    b_mean = results["B"][str(shift)]["mean"]
    gap = d_mean - b_mean
    results["gaps"][str(shift)] = gap
    print(f"    shift={shift:.3f}: D={d_mean:.4f} B={b_mean:.4f} gap={gap:+.4f}")

# Detection floor: lowest shift where D still beats B
d_above_b = [s for s in SHIFTS if results["gaps"][str(s)] > 0]
floor_shift = min(d_above_b) if d_above_b else None
results["analysis"] = {
    "detection_floor": floor_shift,
    "d_above_b_at_all_shifts": len(d_above_b) == len(SHIFTS),
    "gap_trend": "widening" if results["gaps"][str(SHIFTS[-1])] > results["gaps"][str(SHIFTS[0])] else "narrowing",
    "weakest_tested": SHIFTS[0],
}
print(f"\n  ANALYSIS:")
print(f"    D outperforms B at all shifts: {results['analysis']['d_above_b_at_all_shifts']}")
print(f"    Detection floor: shift <= {floor_shift}")
print(f"    Gap trend: {results['analysis']['gap_trend']}")

# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

x = [str(s) for s in SHIFTS]
d_means = [results["D"][str(s)]["mean"] for s in SHIFTS]
b_means = [results["B"][str(s)]["mean"] for s in SHIFTS]
d_stds  = [results["D"][str(s)]["std"]  for s in SHIFTS]
b_stds  = [results["B"][str(s)]["std"]  for s in SHIFTS]
gaps    = [results["gaps"][str(s)]       for s in SHIFTS]

col_d = "#00bfff"
col_b = "#ff6b6b"

# Panel A: Head-to-head bar chart
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor("#1a1a1a")
xpos = np.arange(len(SHIFTS))
w = 0.35
bars_d = ax_a.bar(xpos - w / 2, d_means, w, color=col_d, alpha=0.85, label="Config D")
bars_b = ax_a.bar(xpos + w / 2, b_means, w, color=col_b, alpha=0.85, label="Config B")
ax_a.errorbar(xpos - w / 2, d_means, yerr=d_stds, fmt="none", color="white", capsize=3, linewidth=1)
ax_a.errorbar(xpos + w / 2, b_means, yerr=b_stds, fmt="none", color="white", capsize=3, linewidth=1)
ax_a.set_xticks(xpos)
ax_a.set_xticklabels([str(s) for s in SHIFTS])
ax_a.set_title("Config D vs B — Detection by Signal Strength", color="white", fontsize=10, pad=8)
ax_a.set_xlabel("Precursor Shift (mean_shift)", color="#aaaaaa", fontsize=9)
ax_a.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_a.tick_params(colors="white")
ax_a.legend(fontsize=8, facecolor="#222222", labelcolor="white", framealpha=0.7)
for spine in ax_a.spines.values():
    spine.set_edgecolor("#444444")

# Panel B: D-vs-B gap across shifts
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor("#1a1a1a")
gap_colors = ["#50fa7b" if g > 0 else "#ff5555" for g in gaps]
ax_b.bar(x, gaps, color=gap_colors, alpha=0.85, width=0.5)
ax_b.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
ax_b.set_title("D − B Gap at Each Signal Level", color="white", fontsize=10, pad=8)
ax_b.set_xlabel("Precursor Shift", color="#aaaaaa", fontsize=9)
ax_b.set_ylabel("D − B (prec_det)", color="#aaaaaa", fontsize=9)
ax_b.tick_params(colors="white")
for spine in ax_b.spines.values():
    spine.set_edgecolor("#444444")
for i, (val, lbl) in enumerate(zip(gaps, x)):
    ax_b.text(i, val + (0.001 if val >= 0 else -0.004),
              f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
              color="white", fontsize=8)

# Panel C: Config D learning curves
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor("#1a1a1a")
shift_colors = plt.cm.cool(np.linspace(0.1, 0.9, len(SHIFTS)))
for i, shift in enumerate(SHIFTS):
    c = smooth(curves_D[shift], w=4)
    ax_c.plot(c, color=shift_colors[i], alpha=0.85, linewidth=1.4,
              label=f"shift={shift}")
ax_c.set_title("Config D — Learning Curves by Signal Strength", color="white", fontsize=10, pad=8)
ax_c.set_xlabel("Episode (smoothed)", color="#aaaaaa", fontsize=9)
ax_c.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_c.tick_params(colors="white")
ax_c.legend(fontsize=7, facecolor="#222222", labelcolor="white", framealpha=0.7)
for spine in ax_c.spines.values():
    spine.set_edgecolor("#444444")

# Panel D: Config B learning curves
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor("#1a1a1a")
for i, shift in enumerate(SHIFTS):
    c = smooth(curves_B[shift], w=4)
    ax_d.plot(c, color=shift_colors[i], alpha=0.85, linewidth=1.4,
              label=f"shift={shift}")
ax_d.set_title("Config B — Learning Curves by Signal Strength", color="white", fontsize=10, pad=8)
ax_d.set_xlabel("Episode (smoothed)", color="#aaaaaa", fontsize=9)
ax_d.set_ylabel("Precursor Detection", color="#aaaaaa", fontsize=9)
ax_d.tick_params(colors="white")
ax_d.legend(fontsize=7, facecolor="#222222", labelcolor="white", framealpha=0.7)
for spine in ax_d.spines.values():
    spine.set_edgecolor("#444444")

floor_str = f"D floor ≤ {floor_shift}" if floor_shift else "floor not yet found"
trend_str = results["analysis"]["gap_trend"].upper()
fig.suptitle(
    f"EXP 7 — Signal Detection Floor  |  window=25  |  3 seeds\n"
    f"{floor_str}  |  Gap trend: {trend_str}  |  D above B at all tested shifts: {results['analysis']['d_above_b_at_all_shifts']}",
    color="white", fontsize=11, y=0.98
)

fig_path = os.path.join(OUT_DIR, "exp7_detection_floor.png")
plt.savefig(fig_path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"\n  Figure saved: {fig_path}")

json_path = os.path.join(OUT_DIR, "exp7_summary.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  JSON saved:  {json_path}")
print("\nEXP 7 COMPLETE.")
