# v4 — Phase 3 Language Model

## Version
**v4 — Phase 3 Language Model**

## Date
April 11, 2026 (written; run pending after v3)

## Purpose
Test whether the surprise-gating principle that drove Config D's precursor detection in a 1-D synthetic signal environment transfers to natural language.

The specific hypothesis: if an LM trained to be sensitive to high-surprise / low-expected-by-surface-features inputs, it should perform better on reasoning tasks where the obvious surface-level answer is wrong — i.e., tasks requiring the reader to notice that something slightly unexpected is happening before the explicit question is asked. BIG-Bench Hard is the primary testbed because its tasks were specifically selected to be resistant to simple pattern-matching.

## Hardware
A100 GPU (Google Colab Pro+). Expected runtime: 2–3 hours for full training and evaluation. The bottleneck is GPT-2 fine-tuning on BIG-Bench Hard, not the routing/arbiter logic.

## Base Model
**GPT-2 (124M parameters)** — `gpt2` via HuggingFace Transformers.

This choice is intentional and limited. GPT-2 is small enough to fine-tune in a Colab session without quantization. The results here are directional only (see Honest Caveat below).

**Phase 4A option:** Mistral-7B + LoRA (`mistralai/Mistral-7B-v0.1`, 4-bit QLoRA). This is the real test of the language hypothesis and requires a larger GPU budget or a multi-GPU setup.

---

## What's New

### Architecture: Logic Core + Chaos Core + Arbiter

**Logic Core (LC):** Standard GPT-2 fine-tuned on BIG-Bench Hard tasks using conventional supervised fine-tuning. This is the baseline LM. It learns the expected surface-level patterns.

**Chaos Core (CC):** GPT-2 fine-tuned with a modified training objective. The CC training corpus is seeded by Phase 2 surprise traces: steps where Config D showed high prediction error (top quartile of `pred_error` in `surprise_traces_phase2.json`) are mapped to prompt modifiers that flag misleading surface appearances. Concretely: for BIG-Bench Hard examples where the answer contradicts what a naive pattern-matcher would predict, those examples are upweighted or prepended with a learned `[SURPRISE]` token. The CC is trained to perform better on exactly these examples.

**Arbiter:** A small router (3-layer MLP + softmax) that takes the concatenated hidden states of LC and CC at the final token position and outputs routing probabilities: P(route to LC) and P(route to CC). The Arbiter is trained jointly with a routing loss that maximizes expected accuracy: it learns to route to CC when CC is likely to be correct and LC is likely to be wrong.

The combined system is called GFP (Gut Feeling Parameter) — it uses the CC when the Arbiter detects that the surface-level answer is probably misleading.

### BIG-Bench Hard Evaluation
The script evaluates all three systems (LC, CC, GFP) on the full BIG-Bench Hard task suite using exact-match accuracy. Additional breakdown:
- Per-task accuracy for all 23 BBH tasks
- **Disagreement analysis:** on examples where LC and CC give different predictions, what is GFP's accuracy vs. LC alone? This is the core diagnostic: if GFP > LC on disagreements, the Arbiter is correctly routing to CC on the right examples.
- Disagreement rate: what fraction of examples trigger CC routing?

---

## Key Design: Connection to Phase 2

The CC training corpus construction is the direct mechanistic link between the signal experiment and the language experiment:

1. `surprise_traces_phase2.json` contains step-level `(pred_error, state_label)` pairs for Config D across all seeds.
2. High-surprise events (top quartile `pred_error`) that occurred during **precursor** steps (not during threat, not during normal) are extracted. These are the moments where Config D's prediction model detected something anomalous before the environment explicitly labeled it as a problem.
3. A mapping function converts these to BIG-Bench Hard prompt modifiers: BIG-Bench examples where the answer is non-obvious (surface-level pattern predicts wrong) are analogous to precursor steps in the signal domain.
4. The CC is trained to perform well on these "non-obvious" examples, using a curriculum that starts with the highest-contrast cases (largest gap between obvious-wrong-answer and correct-answer).

The question being asked: **does the mechanism that made Config D sensitive to sub-threshold signals in a 1-D environment translate to a language model being better at tasks where the obvious answer is wrong?**

---

## Status
Script complete at `training/v4_phase3_lm/phase3_chaos_core_lm.py`. Run pending. Requires `surprise_traces_phase2.json` from the v3 run before execution.

To verify the script is importable before running:
```bash
python -c "import phase3_chaos_core_lm; print('OK')"
```

---

## Failure Modes and Fixes

**Arbiter always routes to LC (never uses CC):**
Force balanced routing batches during Arbiter training: ensure that 50% of training examples in the Arbiter optimization step have ground-truth CC-correct / LC-wrong labels. Without this, the Arbiter may learn that LC is right most of the time and collapse to always routing to LC.

**CC reward too sparse (CC doesn't improve over LC on non-obvious examples):**
Lower `threshold_percentile` from 75 to 60 when extracting high-surprise events from Phase 2 traces. This expands the set of examples treated as "non-obvious" in the CC corpus, giving the CC more gradient signal during fine-tuning.

**GFP < LC overall (routing to CC hurts on standard examples):**
Lower `cc_route_min_confidence` — require the Arbiter to be more confident before routing to CC. This reduces false-positive routing (sending standard examples to CC when LC would be correct). Alternatively, add an explicit accuracy penalty to the Arbiter loss for routing to CC on LC-correct examples.

**`surprise_traces_phase2.json` not found:**
The script will raise a `FileNotFoundError` with a clear message pointing to the v3 run. Do not proceed without this file — the CC curriculum construction depends on it.

---

## Success Criteria
Any one of the following is sufficient to treat Phase 3 as a positive result:
- GFP accuracy ≥ LC accuracy overall on BIG-Bench Hard
- CC accuracy > LC accuracy on any single BBH subtask
- GFP accuracy > LC accuracy specifically on examples where LC and CC disagree
- Disagreement rate > 15% (CC is being invoked on a non-trivial fraction of examples)

Note: all four criteria being satisfied simultaneously at GPT-2 scale would be a strong result. Partial confirmation (1–2 criteria) is the expected outcome at this scale.

---

## Honest Caveat
GPT-2 (124M) is small enough that the absolute performance numbers on BIG-Bench Hard will be low. Most BBH tasks require reasoning capabilities that GPT-2 struggles with regardless of fine-tuning strategy. The results at this scale are **directional only** — they test whether the architectural mechanism (surprise-gated routing) can be implemented and whether there is any signal.

**The real test of the language hypothesis is Phase 4A (Mistral-7B + LoRA).** At 7B parameters with instruction-following capability, BBH performance is high enough that the delta between LC, CC, and GFP is interpretable. Phase 3 at GPT-2 scale is a proof-of-concept for the architecture, not a definitive test of the hypothesis.

---

## Required Inputs
- `training/v3_phase2_gpu/surprise_traces_phase2.json` — from v3 run
- BIG-Bench Hard dataset (auto-downloaded by script via HuggingFace datasets)
- GPT-2 weights (auto-downloaded via `transformers`)
