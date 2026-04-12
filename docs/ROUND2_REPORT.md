# GFP Round 2 — Small-Scale Experiment Report

**Date:** April 11, 2026
**Project:** Necessity Labs / Aletheia Framework — Gut Feeling Parameter
**Scale:** MLP FastAgent, hidden=24, window=25 (corrected from Round 1 Exp 3), 3 seeds (42, 43, 44)
**Architecture:** No LSTM. `nn.Linear(window→hidden→hidden)` with Welford online surprise gate.

---

## Background

Round 1 identified five pilot findings. Three remained open:

1. Whether the Welford gate's sensitivity to `std_mult` is suppressed early and only emerges after the running statistics stabilize.
2. Whether Config D's detection advantage over Config B persists below the weakest signal tested in Round 1 (`mean_shift=0.02`).
3. Whether inter-episode pool pre-population — as opposed to the intra-episode sharing in Round 1 Exp 5 — produces a measurable contagion benefit.

All three experiments used `window=25` (updated from the Round 1 default of 50, per Exp 3 result).

---

## EXP 6 — Welford Stabilization Lag

**Hypothesis:** The `std_mult` threshold parameter is ineffective in the short-episode regime because Welford running statistics have not yet converged. As episode count increases, the gate becomes better calibrated and performance should diverge across multiplier values.

**Design:** `std_mult` sweep over [0.25, 0.5, 1.0, 1.5, 2.0], Config D only, compared at 30 episodes (short) vs. 150 episodes (long). 3 seeds, `window=25`.

### Results

| std_mult | Short (30ep) | Long (150ep) | Δ (Long − Short) |
|----------|-------------|--------------|------------------|
| 0.25     | 0.6452      | 0.8551       | +0.2099          |
| 0.50     | 0.6408      | 0.8496       | +0.2088          |
| 1.00     | 0.6208      | 0.8257       | +0.2049          |
| 1.50     | 0.6102      | 0.8106       | +0.2004          |
| 2.00     | 0.5908      | 0.7961       | +0.2053          |

Score range (max − min across std_mults):
- Short regime: **0.0544**
- Long regime: **0.0590**
- Ratio: 1.08× (hypothesis required ≥1.5×)

### Findings

The hypothesis was **not supported**. The monotone ordering — lower std_mult consistently outperforms higher std_mult — was already fully present at 30 episodes. Going from 30 to 150 episodes improved all conditions by roughly equal amounts (~0.20), so the range between conditions barely widened.

This means the Welford gate is discriminative before its statistics have formally converged, at least at the `window=25`, 400-step scale. The mechanism appears to be that a more permissive gate (lower multiplier) produces more frequent gradient updates, which accelerates learning independently of whether the running mean has fully settled.

A secondary finding is that **lower std_mult is consistently better** across both regimes. The optimal multiplier in this setup appears to be ≤0.25. This is counter-intuitive: a stricter gate was expected to produce cleaner signals, but in practice it starves the agent of training signal early in learning.

**Caution:** Seed variance at std_mult=1.5 in the long regime (±0.064) is notably high, suggesting instability at intermediate threshold values in the long-run. This warrants a targeted follow-up.

**Follow-up question:** Is the monotone gate effect a property of the MLP architecture specifically, or does it persist with LSTM? The LSTM's recurrent context may make it less sensitive to observation window thresholds.

---

## EXP 7 — Signal Detection Floor

**Hypothesis:** Config D's detection advantage over Config B persists below `mean_shift=0.02`. The true detection floor — where D's advantage collapses toward chance — has not yet been found.

**Design:** `mean_shift` sweep over [0.005, 0.01, 0.015, 0.02], Config D vs. Config B, 40 episodes, 3 seeds, `window=25`.

### Results

| mean_shift | Config D | Config B | Gap (D−B) |
|------------|----------|----------|-----------|
| 0.005      | 0.6363   | 0.4512   | +0.1851   |
| 0.010      | 0.6353   | 0.4520   | +0.1833   |
| 0.015      | 0.6382   | 0.4526   | +0.1856   |
| 0.020      | 0.6401   | 0.4528   | +0.1873   |

D outperformed B at all four tested shift levels. Gap range across all shifts: ~0.004 (non-monotonic and within noise).

### Findings

The detection floor was **not found within this range**. Config D's advantage over Config B held at every tested level, including `mean_shift=0.005` — which represents a precursor signal one-sixteenth the magnitude of the threat signal (0.35).

Two independent observations complicate interpretation:

**Config B is near or below chance.** B's mean of ~0.451 is below the 0.5 baseline, and Seed 44 scored as low as 0.356. This suggests Config B has not learned meaningful precursor detection at 40 episodes under this signal regime — it may be that B's false-alarm penalty (−0.3) is preventing exploration entirely. If B has collapsed to always predicting no-precursor, the D-vs-B gap is measuring D's absolute performance, not a differential sensitivity advantage.

**Signal strength had almost no effect on either config.** Across a 4× range of shift values, detection scores moved by less than 0.005 per config. Both agents appear insensitive to precursor magnitude at this scale. This is consistent with the Exp 3 finding that the window size (temporal context) matters more than signal magnitude.

**Follow-up questions:**
- What is Config B's actual action distribution at these shift levels? Is it truly exploring or has it collapsed?
- Is the detection floor below 0.005, or does D's performance eventually degrade as noise increasingly swamps the signal? Testing at 0.001 and 0.002 would clarify this.
- Does training longer (150 ep, as in Exp 6's long regime) change B's behavior at weak signals?

---

## EXP 8 — Inter-Episode Contagion (Redesign)

**Hypothesis:** A second agent (A2) that begins training with a surprise pool pre-populated by a first agent (A1) will converge faster and/or achieve higher final performance than a control agent (A3) with no pool access.

**Design:** A1 pre-trains for 50 episodes, writing all surprise events to a shared pool (cap=500). Pool is then frozen. A2 trains for 60 episodes with read-only access to the frozen pool. A3 trains for 60 episodes with no pool. Comparison: convergence speed (episode at which 5-episode rolling mean ≥ 0.40) and final performance (last 8 episodes).

### Results

| Seed | Pool Size | A2 Final | A3 Final | Lift (A2−A3) |
|------|-----------|----------|----------|--------------|
| 42   | 500       | 0.7700   | 0.6464   | +0.1236      |
| 43   | 500       | 0.8022   | 0.6209   | +0.1813      |
| 44   | 500       | 0.8805   | 0.6296   | +0.2509      |
| **Mean** | **500** | **0.8175** | **0.6323** | **+0.1853** |
| Std  | —         | ±0.0464  | ±0.0106  | ±0.0520      |

Convergence speed: Both A2 and A3 reached the 0.40 threshold by episode 5 in all seeds. No convergence speedup was observed.

### Findings

The hypothesis was **partially supported**. The inter-episode redesign produced a clear and consistent +18.5 percentage-point lift in final performance (mean +0.1853) across all three seeds — a direct reversal of the Round 1 Exp 5 result, which showed no benefit.

The critical structural difference: in Round 1, pool entries were written and read within the same episode, meaning Agent 2 was querying a sparse pool at exactly the point when it most needed information. In this redesign, Agent 2 starts from episode 0 with a full 500-entry pool already encoding A1's surprise patterns. The timing of information transfer was the bottleneck, not the pool mechanism itself.

However, **no convergence speedup was observed** — both A2 and A3 crossed the 0.40 threshold by episode 5 regardless of pool access. The benefit of the pre-populated pool appears entirely in asymptotic performance quality, not in the rate of initial learning. This suggests the pool provides guidance that is only usable once A2's own representations have partially matured, not a shortcut that bypasses early training.

A secondary observation: **A3 (control) is low-variance** (±0.0106) while **A2 is high-variance** (±0.0464). The pool introduces a seed-dependent advantage, meaning the value of borrowed context depends on how well A1's learned representations transfer to A2's trajectory. This may be a function of the degree to which the two agents share effective hidden-state geometry.

**Follow-up questions:**
- What pool size is sufficient? All seeds fully saturated the 500-entry cap. Would a smaller, curated pool (high-confidence entries only) perform better than a full pool?
- Does the lift survive when A1 and A2 use different random seeds for the environment as well as initialization? Current design seeds them differently but the environment distribution is the same.
- Can the pool be populated more efficiently — e.g., by seeding it with synthetic surprise events derived from known precursor statistics?

---

## Cross-Experiment Summary

| Experiment | Hypothesis | Key Finding |
|---|---|---|
| EXP 6 — Welford Lag | Not supported | Gate is discriminative before Welford convergence. Monotone: lower std_mult = more training signal = better performance. Effect size: 0.054 short → 0.059 long (1.08×, not 1.5× threshold). |
| EXP 7 — Detection Floor | Floor not found | D outperforms B at all tested levels down to mean_shift=0.005. Gap is flat (~+0.185) suggesting D's advantage is not signal-magnitude dependent. B may have collapsed. |
| EXP 8 — Contagion Redesign | Supported (partially) | +18.5pp lift from pre-populated pool. No convergence speedup. Inter-episode timing was the key fix from Round 1. Pool benefit is asymptotic, not early. |

### What changed from Round 1

- **window=25** used throughout (vs. 50 in Round 1 pilots). Performance levels in EXP 6 and 7 are notably higher than Round 1, consistent with the Exp 3 finding.
- All three Round 2 experiments ran to completion without errors.
- EXP 8 represents the most concrete positive finding of the project so far: a mechanism that reliably improves performance by ~18 percentage points using borrowed experience.

### Remaining open questions for Round 3 or Phase 2

1. **What is Config B's actual behavioral collapse mode at weak signals?** Action distribution logging needed.
2. **Is std_mult=0.25 (or lower) the true optimum?** A sweep from 0.1 to 0.5 in finer steps would locate the effective lower bound.
3. **Does the pool benefit hold at the Phase 2 (A100, 200-episode) scale?** The sandbox pilots use 30–60 episodes; the effect may amplify or diminish at scale.
4. **What is the minimum pool size for the contagion lift?** Testing pool caps of [50, 100, 200, 500] would give a data-efficient transfer curve.
5. **Can A2's high variance be reduced?** Filtering pool entries by minimum surprise magnitude before sharing may improve consistency.

---

*Scale note: All Round 2 results are small-scale CPU pilots (MLP, hidden=24, 30–150 episodes × 400 steps × 3 seeds). Findings are preliminary and directional. Conclusions are held tentatively pending Phase 2 GPU replication.*
