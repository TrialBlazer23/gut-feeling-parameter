"""
scripts/validate_experiment.py
Gut Feeling Parameter — Pre-commit Validation
Necessity Labs / TrialBlazer23

Run this before committing any changes to the Phase 2 experiment script.
A passing result means the core mechanism and required metrics are intact.
This script runs on CPU — no GPU required.

Usage:
    python scripts/validate_experiment.py

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

import sys
import math
import inspect
import importlib.util
import traceback
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"

failures = []
warnings = []

def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    if condition:
        print(f"{PASS}  {name}")
    else:
        tag = WARN if warn_only else FAIL
        print(f"{tag}  {name}")
        if detail:
            print(f"       → {detail}")
        if warn_only:
            warnings.append(name)
        else:
            failures.append(name)


def load_experiment_module():
    """Load the Phase 2 experiment script as a module."""
    script_path = Path("training/v3_phase2_gpu/phase2_gpu_experiment.py")
    if not script_path.exists():
        print(f"{FAIL}  Cannot find experiment script at {script_path}")
        print("       → Run this script from the repository root.")
        sys.exit(1)

    # Strip Colab-specific lines before importing
    source = script_path.read_text()
    cleaned_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        # Skip pip install magic commands
        if stripped.startswith("!pip") or stripped.startswith("get_ipython"):
            cleaned_lines.append("pass  # colab magic removed")
        else:
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    spec = importlib.util.spec_from_loader("phase2", loader=None)
    module = type(sys)("phase2")
    try:
        exec(compile(cleaned, str(script_path), "exec"), module.__dict__)
    except Exception as e:
        print(f"{FAIL}  Experiment script has import/syntax errors:")
        print(f"       → {e}")
        traceback.print_exc()
        sys.exit(1)
    return module


# ── Section 1: File Structure ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 1 — File Structure")
print("=" * 60)

required_files = [
    "training/v3_phase2_gpu/phase2_gpu_experiment.py",
    "docs/HYPOTHESIS.md",
    "docs/EXPERIMENT_LOG.md",
    "AGENTS.md",
    "MEMORY.md",
    "README.md",
    "requirements.txt",
]

for f in required_files:
    check(f"File exists: {f}", Path(f).exists(),
          detail=f"Expected at repo root or relative path shown")

# Phase 1 preservation
check(
    "Phase 1 results directory present",
    Path("results/phase1").exists(),
    detail="results/phase1/ should exist — Phase 1 is completed research",
    warn_only=True
)

# ── Section 2: Load Module ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2 — Module Load")
print("=" * 60)

mod = load_experiment_module()
print(f"{PASS}  Experiment script loads without errors")

# ── Section 3: Required Classes ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3 — Required Agent Classes")
print("=" * 60)

for cls_name in ["ConfigA_Agent", "ConfigD_Agent", "ConfigE_Agent",
                  "RandomAgent", "SignalEnvironment", "EpisodicMemory"]:
    has_cls = hasattr(mod, cls_name)
    check(f"Class defined: {cls_name}", has_cls,
          detail=f"{cls_name} not found in experiment script")

# ── Section 4: Welford M2 Implementation ─────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4 — Welford Algorithm (M2-based)")
print("=" * 60)

# Check for M2 buffer in ConfigD_Agent source
if hasattr(mod, "ConfigD_Agent"):
    source_d = inspect.getsource(mod.ConfigD_Agent)
    has_m2 = "_err_M2" in source_d
    has_rolling_var = "_err_var" in source_d and "/ n" in source_d and "_err_M2" not in source_d

    check(
        "ConfigD_Agent uses M2-based Welford (not rolling variance)",
        has_m2 and not has_rolling_var,
        detail=(
            "Found rolling variance update instead of M2 tracking. "
            "Replace with: self._err_M2 += delta * delta2. "
            "Std = sqrt(M2 / (n-1)). Must match gfp_shared_r2.py."
        )
    )

    # Verify numerical correctness of the Welford implementation
    if has_m2:
        import torch
        import random as _random

        try:
            env_cfg_cls = getattr(mod, "EnvironmentConfig", None)
            exp_cfg_cls = getattr(mod, "ExperimentConfig", None)
            if env_cfg_cls and exp_cfg_cls:
                env_cfg = env_cfg_cls()
                exp_cfg = exp_cfg_cls()
                exp_cfg.n_episodes = 1
                exp_cfg.hidden_size = 32
                exp_cfg.n_layers = 1

                agent = mod.ConfigD_Agent(env_cfg.window_size, exp_cfg)
                # Feed 100 errors and compare Welford std to numpy std
                import numpy as np
                _random.seed(0)
                np.random.seed(0)
                errors = np.random.exponential(0.5, 100).tolist()
                for e in errors:
                    agent._update_running_stats(e)
                welford_std = agent._get_std() if hasattr(agent, "_get_std") else None
                numpy_std = float(np.std(errors, ddof=1))

                if welford_std is not None:
                    relative_error = abs(welford_std - numpy_std) / (numpy_std + 1e-10)
                    check(
                        f"Welford std numerically correct (err={relative_error:.4f}, target<0.01)",
                        relative_error < 0.01,
                        detail=f"Welford={welford_std:.4f}, numpy={numpy_std:.4f}"
                    )
                else:
                    check(
                        "ConfigD_Agent has _get_std() method",
                        False,
                        detail="Add a _get_std() method that returns sqrt(M2 / (n-1))"
                    )
        except Exception as e:
            check("Welford numerical validation", False, detail=str(e))

# ── Section 5: Required Metric Keys ───────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5 — Required Metric Keys in Episode Output")
print("=" * 60)

# Run a minimal 1-episode smoke test to check metric keys
required_metrics = [
    "precursor_det_rate",    # recall
    "precursor_precision",   # precision — must be present
    "precursor_f1",          # f1 — must be present
    "threat_det_rate",
    "action_rate",
    "surprise_rate",
    "mean_pred_error",
    "nonzero_reward_rate",
    "n_surprise_events",
    "surprise_events",
    "hidden_states",
    "true_labels",
    "precursor_base_rate",   # required for enrichment ratio computation
]

try:
    import torch
    import torch.optim as optim

    env_cfg = mod.EnvironmentConfig()
    exp_cfg = mod.ExperimentConfig()
    exp_cfg.n_episodes = 1
    exp_cfg.hidden_size = 32
    exp_cfg.n_layers = 1
    exp_cfg.n_seeds = [42]

    env = mod.SignalEnvironment(env_cfg, seed=42)
    agent_d = mod.ConfigD_Agent(env_cfg.window_size, exp_cfg)
    optimizer_d = optim.Adam(agent_d.parameters(), lr=1e-3)

    # Run one supervised episode (Config A)
    if hasattr(mod, "run_episode_supervised") and hasattr(mod, "ConfigA_Agent"):
        agent_a = mod.ConfigA_Agent(env_cfg.window_size, exp_cfg)
        optimizer_a = optim.Adam(agent_a.parameters(), lr=1e-3)
        metrics_a = mod.run_episode_supervised(agent_a, env, optimizer_a, exp_cfg, train=True)
        for key in required_metrics:
            present = key in metrics_a
            check(
                f"Config A metric present: {key}",
                present,
                detail=f"Key '{key}' missing from run_episode_supervised() output dict"
            )
    else:
        check("run_episode_supervised() exists", False,
              detail="Required for Config A training loop")
        check("ConfigA_Agent exists", hasattr(mod, "ConfigA_Agent"),
              detail="Supervised baseline required for H1 testing")

    # Run one RL episode (Config D)
    if hasattr(mod, "run_episode"):
        env2 = mod.SignalEnvironment(env_cfg, seed=42)
        metrics_d = mod.run_episode(agent_d, env2, optimizer_d, exp_cfg, "D", train=True)
        for key in required_metrics:
            present = key in metrics_d
            check(
                f"Config D metric present: {key}",
                present,
                detail=f"Key '{key}' missing from run_episode() output dict"
            )
    else:
        check("run_episode() exists", False, detail="Required for Config D and E training")

except Exception as e:
    check("Smoke test (1 episode, CPU)", False, detail=str(e))
    traceback.print_exc()

# ── Section 6: Environment Sanity ────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6 — Environment Sanity")
print("=" * 60)

try:
    import numpy as np
    env_cfg = mod.EnvironmentConfig()
    env = mod.SignalEnvironment(env_cfg, seed=42)
    obs, labels = env.reset()

    check("Environment resets without error", True)
    check(
        f"Observation shape correct (window_size={env_cfg.window_size})",
        len(obs) == env_cfg.window_size,
        detail=f"Got obs length {len(obs)}, expected {env_cfg.window_size}"
    )
    check(
        "Labels array length matches n_steps",
        len(labels) == env_cfg.n_steps,
        detail=f"Got {len(labels)} labels, expected {env_cfg.n_steps}"
    )

    n_precursor = int((labels == 1).sum())
    n_threat = int((labels == 2).sum())
    precursor_pct = n_precursor / env_cfg.n_steps * 100
    threat_pct = n_threat / env_cfg.n_steps * 100

    check(
        f"Precursor states present (got {n_precursor}, {precursor_pct:.1f}% of steps)",
        n_precursor > 0,
        detail="Environment generated no precursor states — check precursor_prob config"
    )
    check(
        f"Threat states present (got {n_threat}, {threat_pct:.1f}% of steps)",
        n_threat > 0,
        detail="Environment generated no threat states"
    )
    check(
        "Precursor base rate is non-trivial (1% – 20%)",
        0.01 <= precursor_pct / 100 <= 0.20,
        detail=(
            f"Precursor base rate is {precursor_pct:.1f}%. "
            "Outside 1-20% range makes enrichment ratio hard to interpret."
        ),
        warn_only=True
    )

    # Step through a few steps
    step_labels = []
    for _ in range(100):
        next_obs, true_label, done = env.step(0)
        step_labels.append(int(true_label))
        if done:
            break

    check(
        "Environment steps without error",
        True
    )
    check(
        "Step labels are valid (0-3)",
        all(0 <= l <= 3 for l in step_labels),
        detail=f"Found labels outside 0-3 range: {set(step_labels) - {0,1,2,3}}"
    )

except Exception as e:
    check("Environment sanity checks", False, detail=str(e))
    traceback.print_exc()

# ── Section 7: Random Baseline ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 7 — Random Baseline")
print("=" * 60)

check(
    "evaluate_random_baseline() function exists",
    hasattr(mod, "evaluate_random_baseline"),
    detail="Random baseline required as confound floor — add evaluate_random_baseline()"
)
check(
    "RandomAgent class exists",
    hasattr(mod, "RandomAgent"),
    detail="RandomAgent required — uniform action sampling, no learning"
)

if hasattr(mod, "evaluate_random_baseline") and hasattr(mod, "RandomAgent"):
    try:
        env_cfg = mod.EnvironmentConfig()
        env = mod.SignalEnvironment(env_cfg, seed=0)
        result = mod.evaluate_random_baseline(env, n_episodes=5, seed=99)
        check(
            "Random baseline runs without error",
            True
        )
        check(
            "Random baseline precursor recall is between 0.3 and 0.7 (sanity)",
            0.3 <= result.get("precursor_det_rate_mean", 0) <= 0.7,
            detail=(
                f"Got {result.get('precursor_det_rate_mean', 'N/A'):.3f}. "
                "Random agent should get ~0.5 recall if action_rate ≈ 0.5."
            ),
            warn_only=True
        )
        print(f"         Random recall: {result.get('precursor_det_rate_mean', 0):.3f}  "
              f"action_rate: {result.get('action_rate_mean', 0):.3f}")
    except Exception as e:
        check("Random baseline smoke test", False, detail=str(e))

# ── Section 8: Statistical Analysis Functions ─────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 8 — Statistical Analysis")
print("=" * 60)

check(
    "cohens_d() function exists",
    hasattr(mod, "cohens_d"),
    detail="Required for effect size reporting"
)
check(
    "get_final_n_ep_mean() function exists",
    hasattr(mod, "get_final_n_ep_mean"),
    detail="Required for per-seed metric aggregation"
)

# Check that D vs A comparison is present in Cell 7 source
script_text = Path("training/v3_phase2_gpu/phase2_gpu_experiment.py").read_text()
check(
    "D vs A comparison present in script",
    '"D_vs_A"' in script_text or "D vs A" in script_text or "D_vs_A" in script_text,
    detail=(
        "Could not find D vs A comparison string in script. "
        "Primary statistical comparison must be Config D vs Config A."
    )
)
check(
    "Enrichment ratio computed or referenced",
    "enrichment" in script_text.lower(),
    detail=(
        "Enrichment ratio not found. Required: "
        "(surprise_precursor_pct / precursor_base_rate). Add to Cell 8."
    ),
    warn_only=True
)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

if failures:
    print(f"\n  {len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"    ✗ {f}")
    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    ⚠ {w}")
    print("\n  Do not commit until all failures are resolved.\n")
    sys.exit(1)
elif warnings:
    print(f"\n  All required checks PASSED with {len(warnings)} warning(s):")
    for w in warnings:
        print(f"    ⚠ {w}")
    print("\n  Warnings do not block commit but should be reviewed.\n")
    sys.exit(0)
else:
    print("\n  All checks PASSED. Safe to commit.\n")
    sys.exit(0)
