# Round 2 Results

Small-scale CPU pilots targeting three open questions from Round 1.
Architecture: FastAgent MLP, hidden=24, window=25 (corrected from Round 1), 3 seeds.

## Experiments
- **EXP 6** — Welford Stabilization Lag: std_mult sweep at 30ep vs 150ep
- **EXP 7** — Signal Detection Floor: mean_shift [0.005, 0.01, 0.015, 0.02]
- **EXP 8** — Inter-Episode Contagion Redesign: pre-populated pool

## Key Findings
- EXP 6: Hypothesis not supported. Gate is discriminative before Welford convergence. Lower std_mult consistently better.
- EXP 7: Detection floor not found. D outperforms B at all tested levels including shift=0.005.
- EXP 8: +18.5pp lift from pre-populated inter-episode pool. Timing of sharing is the key variable.

See `docs/ROUND2_REPORT.md` for full written report.
