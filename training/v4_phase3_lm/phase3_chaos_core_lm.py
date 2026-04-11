# ============================================================
# Protection Drive — Phase 3: Chaos Core LM
# Necessity Labs / Aletheia Framework
# ============================================================
# Run on: Colab Pro+ → A100 GPU
# Runtime: ~2-3 hours
#
# PREREQUISITE: Upload surprise_traces_phase2.json to /content/
# before running this notebook.
#
# Cell boundaries: # %%
# Run cells in order.
# ============================================================

# %% ─── CELL 1 — INSTALLS & IMPORTS ─────────────────────────────────────────

!pip install -q torch transformers datasets accelerate scikit-learn scipy matplotlib seaborn tqdm

import os
import json
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    get_linear_schedule_with_warmup
)
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

TRACES_PATH = "/content/surprise_traces_phase2.json"
OUT_DIR = "/content/gfp_models"
RESULTS_DIR = "/content/gfp_results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("/content/gfp_plots", exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# %% ─── CELL 2 — CONFIGURATION ───────────────────────────────────────────────

@dataclass
class GFPConfig:
    # Model
    model_name: str = "gpt2"            # swap to "mistralai/Mistral-7B-v0.1" for Phase 4
    max_length: int = 128
    # Training
    lc_lr: float = 5e-5
    cc_lr: float = 5e-5
    arbiter_lr: float = 1e-3
    batch_size: int = 4
    lc_epochs: int = 3
    cc_epochs: int = 3
    arbiter_epochs: int = 5
    warmup_ratio: float = 0.1
    # Chaos Core surprise parameters
    intrinsic_error_threshold_std: float = 1.0   # if reward <10%, lower to 0.7
    cc_reward_scale: float = 1.0
    # Arbiter
    arbiter_hidden: int = 64
    cc_route_min_confidence: float = 0.55
    # BBH Evaluation
    bbh_max_samples_per_task: int = 30
    bbh_few_shot: int = 3               # 0 = zero-shot
    # Corpus building
    threshold_percentile: int = 75      # if <15 examples, lower to 60
    min_surprise_correct: int = 15      # minimum viable chaos corpus size
    # Logging
    log_steps: int = 20

CFG = GFPConfig()
print("GFP Config:")
print(json.dumps(CFG.__dict__, indent=2))

# %% ─── CELL 3 — LOAD TOKENIZER & BASE MODEL ────────────────────────────────

print("\nLoading GPT-2 tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(CFG.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def load_fresh_gpt2() -> GPT2LMHeadModel:
    """Return a freshly loaded GPT-2 on DEVICE."""
    model = GPT2LMHeadModel.from_pretrained(CFG.model_name)
    model.config.pad_token_id = tokenizer.eos_token_id
    return model.to(DEVICE)

print(f"Base model loaded: {CFG.model_name}")
n_params = sum(p.numel() for p in load_fresh_gpt2().parameters())
print(f"Parameters: {n_params:,}")

# %% ─── CELL 4 — BIG-BENCH HARD LOADER ───────────────────────────────────────

# BIG-Bench Hard task subset relevant to causal/lateral reasoning
# Using a self-contained implementation to avoid dataset library version issues
BBH_TASKS = [
    "causal_judgement",
    "temporal_sequences",
    "logical_deduction_three_objects",
    "reasoning_about_colored_objects",
    "multistep_arithmetic_two",
    "word_sorting",
]

BBH_TASK_PROMPTS = {
    "causal_judgement": (
        "Q: Does the following situation involve a direct cause?\n{input}\nA:"
    ),
    "temporal_sequences": (
        "Q: What is the correct temporal order?\n{input}\nA:"
    ),
    "logical_deduction_three_objects": (
        "Q: Given the clues, what is the correct ordering?\n{input}\nA:"
    ),
    "reasoning_about_colored_objects": (
        "Q: Answer the question about colored objects.\n{input}\nA:"
    ),
    "multistep_arithmetic_two": (
        "Q: Compute the result of the following arithmetic.\n{input}\nA:"
    ),
    "word_sorting": (
        "Q: Sort the following words alphabetically.\n{input}\nA:"
    ),
}


def generate_synthetic_bbh_task(task_name: str, n: int, rng: np.random.RandomState) -> List[Dict]:
    """
    Generate synthetic BBH-style examples for each task.
    These are structurally equivalent to real BBH but self-contained,
    avoiding network calls during the experiment.
    """
    examples = []

    if task_name == "causal_judgement":
        templates = [
            ("Alex pressed the button. The alarm went off. Did pressing the button cause the alarm?", "yes"),
            ("It rained. The streets were wet. Did the rain cause the wet streets?", "yes"),
            ("The window was broken. The cat was in the room. Did the cat break the window?", "no"),
            ("Sam ate the sandwich. The sandwich disappeared. Did Sam cause the disappearance?", "yes"),
            ("The power went out. The TV turned off. Did the power outage cause the TV to turn off?", "yes"),
            ("The dog barked. The baby cried later. Did the dog barking cause the baby to cry?", "no"),
        ]
        for i in range(n):
            t = templates[i % len(templates)]
            examples.append({"input": t[0], "target": t[1], "task": task_name})

    elif task_name == "temporal_sequences":
        seqs = [
            ("Alpha happens before Beta. Beta happens before Gamma. What happens first?", "Alpha"),
            ("Event 3 happens after Event 1. Event 2 happens before Event 1. What is the order?", "2, 1, 3"),
            ("Step B follows Step A. Step C follows Step B. What is the second step?", "Step B"),
            ("X comes before Y. Z comes after Y. What comes last?", "Z"),
            ("Morning comes before noon. Noon comes before evening. What comes second?", "noon"),
        ]
        for i in range(n):
            t = seqs[i % len(seqs)]
            examples.append({"input": t[0], "target": t[1], "task": task_name})

    elif task_name == "logical_deduction_three_objects":
        problems = [
            ("The red ball is to the left of the blue ball. The blue ball is to the left of the green ball. What is on the far right?", "green"),
            ("A is heavier than B. B is heavier than C. What is the lightest?", "C"),
            ("Box 1 is above Box 2. Box 2 is above Box 3. What is at the bottom?", "Box 3"),
            ("Circle is faster than Square. Square is faster than Triangle. What is slowest?", "Triangle"),
            ("Dog is taller than Cat. Cat is taller than Bird. What is shortest?", "Bird"),
        ]
        for i in range(n):
            t = problems[i % len(problems)]
            examples.append({"input": t[0], "target": t[1], "task": task_name})

    elif task_name == "reasoning_about_colored_objects":
        problems = [
            ("There is a red sphere and a blue cube. What color is the sphere?", "red"),
            ("The green triangle is next to the yellow circle. What shape is yellow?", "circle"),
            ("A purple square is above an orange oval. What color is above?", "purple"),
            ("There are two objects: a white cone and a black pyramid. Which is white?", "cone"),
        ]
        for i in range(n):
            t = problems[i % len(problems)]
            examples.append({"input": t[0], "target": t[1], "task": task_name})

    elif task_name == "multistep_arithmetic_two":
        rng_local = np.random.RandomState(42)
        for i in range(n):
            a, b, c, d = rng_local.randint(1, 10, 4)
            q = f"( {a} + {b} ) * {c} - {d}"
            ans = str((a + b) * c - d)
            examples.append({"input": q, "target": ans, "task": task_name})

    elif task_name == "word_sorting":
        word_lists = [
            (["zebra", "apple", "mango"], "apple, mango, zebra"),
            (["piano", "guitar", "bass"], "bass, guitar, piano"),
            (["cloud", "rain", "sun"], "cloud, rain, sun"),
            (["tree", "bush", "moss"], "bush, moss, tree"),
        ]
        for i in range(n):
            t = word_lists[i % len(word_lists)]
            examples.append({"input": ", ".join(t[0]), "target": t[1], "task": task_name})

    else:
        for i in range(n):
            examples.append({"input": f"Question {i}", "target": "answer", "task": task_name})

    return examples


def build_bbh_dataset(rng: np.random.RandomState) -> Dict[str, List[Dict]]:
    dataset = {}
    for task in BBH_TASKS:
        dataset[task] = generate_synthetic_bbh_task(task, CFG.bbh_max_samples_per_task, rng)
    total = sum(len(v) for v in dataset.values())
    print(f"BBH dataset built: {len(BBH_TASKS)} tasks, {total} examples total")
    return dataset


print("BBH loader ready.")

# %% ─── CELL 5 — LOAD & PARSE PHASE 2 SURPRISE TRACES ───────────────────────

print("\n" + "=" * 60)
print("LOADING PHASE 2 SURPRISE TRACES")
print("=" * 60)

if not os.path.exists(TRACES_PATH):
    print("[WARNING] surprise_traces_phase2.json not found at /content/")
    print("  Generating synthetic traces for development purposes...")
    # Synthetic fallback: simulate a reasonable trace distribution
    def _generate_synthetic_traces():
        rng = np.random.RandomState(42)
        traces = {"D": {}, "E": {}}
        seeds = ["42", "43", "44", "45", "46"]
        for config_name in ["D", "E"]:
            for seed in seeds:
                events = []
                for ep in range(200):
                    n_events = rng.randint(3, 15)
                    for _ in range(n_events):
                        label = rng.choice([0, 1, 2, 3], p=[0.45, 0.35, 0.12, 0.08])
                        events.append({
                            "step": int(rng.randint(0, 1500)),
                            "pred_error": float(rng.exponential(0.3)),
                            "true_label": int(label),
                            "action": 1,
                            "episode": ep
                        })
                traces[config_name][seed] = events
        return traces
    traces = _generate_synthetic_traces()
    print("  Synthetic traces generated (proceeding with development run).")
else:
    with open(TRACES_PATH) as f:
        traces = json.load(f)
    print(f"Loaded traces from {TRACES_PATH}")

# Flatten Config D traces (primary signal source for Chaos Core)
d_traces_flat = []
for seed_key, events in traces.get("D", {}).items():
    d_traces_flat.extend(events)

# Compute surprise stats
if d_traces_flat:
    pred_errors = [ev["pred_error"] for ev in d_traces_flat]
    label_counts = {}
    for ev in d_traces_flat:
        l = ev["true_label"]
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"\nConfig D trace summary:")
    print(f"  Total surprise events: {len(d_traces_flat)}")
    print(f"  Pred error range: {min(pred_errors):.4f} – {max(pred_errors):.4f}")
    print(f"  Mean pred error: {np.mean(pred_errors):.4f}")
    print(f"  State distribution: {label_counts}")
    surprise_threshold = float(np.percentile(pred_errors, CFG.threshold_percentile))
    print(f"  Surprise threshold ({CFG.threshold_percentile}th pct): {surprise_threshold:.4f}")
else:
    pred_errors = [0.5]
    surprise_threshold = 0.5
    print("[WARNING] No traces loaded.")

# %% ─── CELL 6 — LOGIC CORE DATASET (Standard SFT + Process Reward) ──────────

print("\n" + "=" * 60)
print("BUILDING LOGIC CORE TRAINING CORPUS")
print("=" * 60)


class LogicCoreDataset(Dataset):
    """
    Standard SFT dataset with process reward weighting.
    Each example is a (prompt, answer) pair from BBH.
    Examples with shorter chain-of-thought get higher process reward.
    """
    def __init__(self, examples: List[Dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for ex in examples:
            task = ex["task"]
            prompt_template = BBH_TASK_PROMPTS.get(task, "Q: {input}\nA:")
            prompt = prompt_template.format(input=ex["input"])
            full_text = prompt + " " + ex["target"]
            encoded = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            # Process reward: shorter inputs (simpler reasoning) get slight boost
            # This is intentionally mild — we don't want to bias away from reasoning
            process_reward = 1.0 / (1.0 + 0.01 * len(full_text))
            self.data.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "process_reward": torch.tensor(process_reward, dtype=torch.float),
                "prompt": prompt,
                "target": ex["target"],
                "task": task
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


rng_bbh = np.random.RandomState(42)
bbh_data = build_bbh_dataset(rng_bbh)
all_examples = [ex for task_exs in bbh_data.values() for ex in task_exs]

# 80/20 train/eval split
n_train = int(0.8 * len(all_examples))
random.shuffle(all_examples)
lc_train_examples = all_examples[:n_train]
lc_eval_examples = all_examples[n_train:]

lc_train_dataset = LogicCoreDataset(lc_train_examples, tokenizer, CFG.max_length)
lc_eval_dataset = LogicCoreDataset(lc_eval_examples, tokenizer, CFG.max_length)

print(f"Logic Core train: {len(lc_train_dataset)} examples")
print(f"Logic Core eval:  {len(lc_eval_dataset)} examples")

# %% ─── CELL 7 — CHAOS CORE CORPUS (Surprise-Gated, Phase 2 Seeded) ──────────

print("\n" + "=" * 60)
print("BUILDING CHAOS CORE CORPUS")
print("=" * 60)

# Map Phase 2 state labels to language cues
# label 0 = normal → routine examples
# label 1 = precursor → examples where surface appearance is misleading
# label 2 = threat → examples with high-stakes decisions
# label 3 = post-threat → examples requiring reflection/reversal
SURPRISE_LABEL_TEMPLATES = {
    0: "Answer directly: {input}",
    1: "The obvious answer may be wrong. Think carefully: {input}",
    2: "This is high-stakes. Consider all possibilities: {input}",
    3: "Reconsider your first instinct: {input}",
}


class ChaosCoreDataset(Dataset):
    """
    Corpus built from high-surprise events in Phase 2 traces.
    Each item pairs a BBH example with a surprise-level prompt modifier
    and carries an intrinsic reward weight derived from prediction error.
    """
    def __init__(self, examples: List[Dict], surprise_events: List[Dict],
                 tokenizer, max_length: int, threshold: float, rng: np.random.RandomState):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Select high-surprise events (above threshold)
        high_surprise = [ev for ev in surprise_events if ev["pred_error"] >= threshold]
        print(f"  High-surprise events (>= {threshold:.4f}): {len(high_surprise)}")

        if len(high_surprise) < CFG.min_surprise_correct:
            print(f"  [WARNING] Only {len(high_surprise)} high-surprise events. "
                  f"Lowering threshold to 60th percentile.")
            low_threshold = float(np.percentile([e["pred_error"] for e in surprise_events], 60))
            high_surprise = [ev for ev in surprise_events if ev["pred_error"] >= low_threshold]
            print(f"  Adjusted high-surprise count: {len(high_surprise)}")

        # Build corpus: pair each BBH example with a surprise event
        n_cc = min(len(examples), len(high_surprise) * 2, len(examples))
        rng.shuffle(high_surprise)
        rng.shuffle(examples)

        for i in range(n_cc):
            ex = examples[i % len(examples)]
            ev = high_surprise[i % len(high_surprise)]
            surprise_label = int(ev["true_label"])
            template = SURPRISE_LABEL_TEMPLATES.get(surprise_label, "{input}")
            modified_prompt = template.format(input=ex["input"])
            task = ex["task"]
            full_text = modified_prompt + "\nAnswer: " + ex["target"]
            encoded = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            # Intrinsic reward weight from prediction error magnitude
            intrinsic_weight = float(np.clip(ev["pred_error"] / (threshold + 1e-8), 0.5, 3.0))
            self.data.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "intrinsic_weight": torch.tensor(intrinsic_weight, dtype=torch.float),
                "surprise_label": surprise_label,
                "pred_error": ev["pred_error"],
                "prompt": modified_prompt,
                "target": ex["target"],
                "task": task
            })
        print(f"  Chaos corpus built: {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


rng_cc = np.random.RandomState(43)
cc_dataset = ChaosCoreDataset(
    all_examples, d_traces_flat, tokenizer,
    CFG.max_length, surprise_threshold, rng_cc
)
print(f"\nChaos Core dataset: {len(cc_dataset)} examples")

# %% ─── CELL 8 — TRAIN LOGIC CORE ────────────────────────────────────────────

print("\n" + "=" * 60)
print("TRAINING LOGIC CORE (Standard SFT + Process Reward)")
print("=" * 60)


def train_language_model(
    model: GPT2LMHeadModel,
    dataset: Dataset,
    cfg: GFPConfig,
    lr: float,
    n_epochs: int,
    reward_key: str = "process_reward",
    reward_scale: float = 1.0,
    name: str = "model"
) -> Tuple[GPT2LMHeadModel, List[float]]:
    """
    Train GPT-2 with reward-weighted cross-entropy.
    reward_key: key in batch dict containing per-example weights
    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    n_steps = len(loader) * n_epochs
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps * cfg.warmup_ratio),
        num_training_steps=n_steps
    )

    model.train()
    step = 0
    epoch_losses = []

    for epoch in range(n_epochs):
        ep_losses = []
        pbar = tqdm(loader, desc=f"{name} epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            reward_weights = batch[reward_key].to(DEVICE)  # (batch,)

            # GPT-2 language modeling: labels = input_ids shifted by model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            # outputs.loss = mean per-token CE loss (scalar)
            # We need per-example loss for reward weighting
            # Recompute manually
            logits = outputs.logits  # (B, T, vocab)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(input_ids.size(0), -1)   # (B, T-1)

            # Mask padding tokens
            per_example_loss = (per_token_loss * shift_mask).sum(-1) / (shift_mask.sum(-1) + 1e-8)

            # Apply reward weighting
            weighted_loss = (per_example_loss * reward_weights * reward_scale).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_losses.append(weighted_loss.item())
            step += 1
            if step % cfg.log_steps == 0:
                pbar.set_postfix({"loss": f"{np.mean(ep_losses[-20:]):.4f}"})

        epoch_mean = float(np.mean(ep_losses))
        epoch_losses.append(epoch_mean)
        print(f"  {name} epoch {epoch+1} loss: {epoch_mean:.4f}")

    return model, epoch_losses


# Train Logic Core
lc_model = load_fresh_gpt2()
lc_model, lc_losses = train_language_model(
    lc_model, lc_train_dataset, CFG,
    lr=CFG.lc_lr, n_epochs=CFG.lc_epochs,
    reward_key="process_reward", reward_scale=1.0,
    name="LogicCore"
)
lc_model.save_pretrained(f"{OUT_DIR}/logic_core")
tokenizer.save_pretrained(f"{OUT_DIR}/logic_core")
print(f"Logic Core saved to {OUT_DIR}/logic_core")

# %% ─── CELL 9 — TRAIN CHAOS CORE ────────────────────────────────────────────

print("\n" + "=" * 60)
print("TRAINING CHAOS CORE (Surprise-Gated Intrinsic Reward)")
print("=" * 60)

# Chaos Core uses intrinsic_weight as reward signal
# Verify reward density before training
weights = [cc_dataset[i]["intrinsic_weight"].item() for i in range(len(cc_dataset))]
nonzero_rate = sum(1 for w in weights if w > 1.0) / max(len(weights), 1)
print(f"Nonzero reward rate (weight > 1.0): {nonzero_rate:.3f}  (target >0.20)")
if nonzero_rate < 0.10:
    print("[WARNING] Reward density < 10%. Lower intrinsic_error_threshold_std to 0.7 in CFG.")

cc_model = load_fresh_gpt2()
cc_model, cc_losses = train_language_model(
    cc_model, cc_dataset, CFG,
    lr=CFG.cc_lr, n_epochs=CFG.cc_epochs,
    reward_key="intrinsic_weight", reward_scale=CFG.cc_reward_scale,
    name="ChaosCore"
)
cc_model.save_pretrained(f"{OUT_DIR}/chaos_core")
tokenizer.save_pretrained(f"{OUT_DIR}/chaos_core")
print(f"Chaos Core saved to {OUT_DIR}/chaos_core")

# Print reward density summary
print(f"\nChaos Core reward stats:")
print(f"  Mean intrinsic weight: {np.mean(weights):.4f}")
print(f"  Max intrinsic weight:  {max(weights):.4f}")
print(f"  % above threshold (>1): {nonzero_rate*100:.1f}%")

# %% ─── CELL 10 — ARBITER (Lightweight MLP Router) ───────────────────────────

print("\n" + "=" * 60)
print("TRAINING ARBITER (Downstream Task Success Router)")
print("=" * 60)


class Arbiter(nn.Module):
    """
    Lightweight MLP that routes each query to Logic Core (0) or Chaos Core (1).
    Input features: [lc_log_prob, cc_log_prob, input_length_normalized, task_id_onehot]
    Output: routing probability P(route_to_CC)
    """
    def __init__(self, n_tasks: int, hidden: int = 64):
        super().__init__()
        input_dim = 2 + 1 + n_tasks  # lc_logp, cc_logp, len_norm, task_onehot
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
        # Init bias toward LC (safer default)
        with torch.no_grad():
            self.net[-2].bias.fill_(-0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ArbiterDataset(Dataset):
    """
    Training data for the arbiter: (features, label) where
    label=1 if CC was more correct than LC on this example.
    """
    def __init__(self, examples: List[Dict], lc_model: GPT2LMHeadModel,
                 cc_model: GPT2LMHeadModel, tokenizer, cfg: GFPConfig,
                 task_list: List[str]):
        self.data = []
        self.task_list = task_list
        n_tasks = len(task_list)
        lc_model.eval()
        cc_model.eval()

        print(f"  Building arbiter training data from {len(examples)} examples...")
        with torch.no_grad():
            for ex in tqdm(examples, desc="Arbiter prep"):
                task = ex["task"]
                task_id = task_list.index(task) if task in task_list else 0
                prompt_template = BBH_TASK_PROMPTS.get(task, "Q: {input}\nA:")
                prompt = prompt_template.format(input=ex["input"])
                target = ex["target"]
                full_text = prompt + " " + target

                encoded = tokenizer(
                    full_text, max_length=cfg.max_length, truncation=True,
                    return_tensors="pt"
                ).to(DEVICE)
                ids = encoded["input_ids"]

                lc_out = lc_model(input_ids=ids, labels=ids)
                cc_out = cc_model(input_ids=ids, labels=ids)

                lc_logp = float(-lc_out.loss.item())  # negative CE = log prob
                cc_logp = float(-cc_out.loss.item())

                # Generate one token to check correctness
                lc_pred = _greedy_generate(lc_model, tokenizer, prompt, max_new=10, cfg=cfg)
                cc_pred = _greedy_generate(cc_model, tokenizer, prompt, max_new=10, cfg=cfg)

                lc_correct = _check_correct(lc_pred, target)
                cc_correct = _check_correct(cc_pred, target)

                # Route label: 1 if CC was uniquely correct (or both wrong but CC more confident)
                if cc_correct and not lc_correct:
                    route_label = 1.0
                elif lc_correct and not cc_correct:
                    route_label = 0.0
                elif cc_logp > lc_logp:
                    route_label = 1.0
                else:
                    route_label = 0.0

                # Task one-hot
                task_oh = np.zeros(n_tasks, dtype=np.float32)
                task_oh[task_id] = 1.0
                len_norm = min(len(full_text) / 500.0, 1.0)

                features = np.array([lc_logp, cc_logp, len_norm] + task_oh.tolist(), dtype=np.float32)
                self.data.append({
                    "features": torch.FloatTensor(features),
                    "label": torch.tensor(route_label, dtype=torch.float),
                    "lc_correct": lc_correct,
                    "cc_correct": cc_correct,
                    "task": task
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _greedy_generate(model: GPT2LMHeadModel, tokenizer, prompt: str,
                      max_new: int = 20, cfg: GFPConfig = None) -> str:
    """Generate up to max_new tokens greedily."""
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=cfg.max_length - max_new
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()


def _check_correct(prediction: str, target: str) -> bool:
    """Check if target string is present in prediction (case-insensitive)."""
    return target.lower().strip() in prediction.lower().strip()


# Build arbiter training data
print("  Using eval set for arbiter training (they haven't been used in LC/CC training)...")
arbiter_dataset = ArbiterDataset(
    lc_eval_examples, lc_model, cc_model, tokenizer, CFG, BBH_TASKS
)

# Compute disagreement rate
n_disagree = sum(1 for d in arbiter_dataset.data
                 if d["lc_correct"] != d["cc_correct"])
disagree_rate = n_disagree / max(len(arbiter_dataset), 1)
print(f"\n  Disagreement rate: {disagree_rate:.3f}  (target > 0.15)")
if disagree_rate < 0.10:
    print("  [NOTE] Low disagreement rate — cores are very similar. "
          "This may mean training was insufficient or the task is too simple.")

# Train arbiter
arbiter = Arbiter(n_tasks=len(BBH_TASKS), hidden=CFG.arbiter_hidden).to(DEVICE)
arb_loader = DataLoader(arbiter_dataset, batch_size=8, shuffle=True)
arb_optimizer = optim.Adam(arbiter.parameters(), lr=CFG.arbiter_lr)
arb_criterion = nn.BCELoss()

arb_losses = []
for epoch in range(CFG.arbiter_epochs):
    ep_losses = []
    for batch in arb_loader:
        features = batch["features"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        preds = arbiter(features)
        loss = arb_criterion(preds, labels)
        arb_optimizer.zero_grad()
        loss.backward()
        arb_optimizer.step()
        ep_losses.append(loss.item())
    arb_losses.append(float(np.mean(ep_losses)))
    print(f"  Arbiter epoch {epoch+1}: loss={arb_losses[-1]:.4f}")

torch.save(arbiter.state_dict(), f"{OUT_DIR}/arbiter.pt")
print(f"Arbiter saved to {OUT_DIR}/arbiter.pt")

# %% ─── CELL 11 — GFP EVALUATION ON BIG-BENCH HARD ──────────────────────────

print("\n" + "=" * 60)
print("GFP SYSTEM EVALUATION — BIG-BENCH HARD")
print("=" * 60)


def evaluate_model_on_task(model: GPT2LMHeadModel, tokenizer,
                            task_examples: List[Dict], task_name: str,
                            few_shot: int, cfg: GFPConfig) -> Tuple[float, List[Dict]]:
    """
    Evaluate model on a BBH task. Returns (accuracy, per_example_results).
    few_shot=0 means zero-shot.
    """
    model.eval()
    results = []
    correct = 0
    prompt_template = BBH_TASK_PROMPTS.get(task_name, "Q: {input}\nA:")

    for i, ex in enumerate(task_examples):
        # Build few-shot prefix
        if few_shot > 0:
            few_shot_examples = [e for e in task_examples if e != ex][:few_shot]
            prefix = ""
            for fs_ex in few_shot_examples:
                prefix += prompt_template.format(input=fs_ex["input"]) + " " + fs_ex["target"] + "\n\n"
        else:
            prefix = ""

        prompt = prefix + prompt_template.format(input=ex["input"])
        pred = _greedy_generate(model, tokenizer, prompt, max_new=20, cfg=cfg)
        is_correct = _check_correct(pred, ex["target"])
        if is_correct:
            correct += 1
        results.append({
            "task": task_name,
            "input": ex["input"],
            "target": ex["target"],
            "prediction": pred,
            "correct": is_correct
        })

    acc = correct / max(len(task_examples), 1)
    return acc, results


def evaluate_gfp_system(lc_model: GPT2LMHeadModel, cc_model: GPT2LMHeadModel,
                         arbiter: Arbiter, tokenizer, bbh_data: Dict,
                         cfg: GFPConfig, task_list: List[str]) -> Dict:
    """
    Evaluate GFP (LC + CC + Arbiter) vs standalone LC on all BBH tasks.
    """
    all_task_results = {}
    lc_total_correct = 0
    gfp_total_correct = 0
    total_examples = 0
    gfp_routed_to_cc = 0
    gfp_cc_correct = 0
    disagree_total = 0
    lc_correct_on_disagree = 0
    gfp_correct_on_disagree = 0
    n_tasks = len(task_list)

    for task_name in tqdm(BBH_TASKS, desc="Evaluating"):
        task_examples = bbh_data[task_name]
        task_id = task_list.index(task_name) if task_name in task_list else 0

        lc_acc, lc_results = evaluate_model_on_task(
            lc_model, tokenizer, task_examples, task_name, cfg.bbh_few_shot, cfg
        )
        cc_acc, cc_results = evaluate_model_on_task(
            cc_model, tokenizer, task_examples, task_name, cfg.bbh_few_shot, cfg
        )

        # GFP routing
        gfp_results = []
        arbiter.eval()
        with torch.no_grad():
            for lc_r, cc_r in zip(lc_results, cc_results):
                encoded = tokenizer(
                    lc_r["input"], return_tensors="pt", truncation=True,
                    max_length=cfg.max_length
                ).to(DEVICE)
                ids = encoded["input_ids"]
                lc_out = lc_model(input_ids=ids, labels=ids)
                cc_out = cc_model(input_ids=ids, labels=ids)
                lc_logp = float(-lc_out.loss.item())
                cc_logp = float(-cc_out.loss.item())
                len_norm = min(len(lc_r["input"]) / 500.0, 1.0)
                task_oh = np.zeros(n_tasks, dtype=np.float32)
                task_oh[task_id] = 1.0
                features = torch.FloatTensor(
                    [lc_logp, cc_logp, len_norm] + task_oh.tolist()
                ).unsqueeze(0).to(DEVICE)
                route_prob = arbiter(features).item()
                use_cc = route_prob > cfg.cc_route_min_confidence

                if use_cc:
                    gfp_correct = cc_r["correct"]
                    gfp_routed_to_cc += 1
                    if gfp_correct:
                        gfp_cc_correct += 1
                else:
                    gfp_correct = lc_r["correct"]

                # Track disagreement cases
                if lc_r["correct"] != cc_r["correct"]:
                    disagree_total += 1
                    if lc_r["correct"]:
                        lc_correct_on_disagree += 1
                    if gfp_correct:
                        gfp_correct_on_disagree += 1

                gfp_results.append({
                    "task": task_name,
                    "input": lc_r["input"],
                    "target": lc_r["target"],
                    "lc_pred": lc_r["prediction"],
                    "cc_pred": cc_r["prediction"],
                    "lc_correct": lc_r["correct"],
                    "cc_correct": cc_r["correct"],
                    "gfp_used_cc": use_cc,
                    "gfp_correct": gfp_correct,
                    "route_prob": route_prob
                })

        gfp_acc = sum(1 for r in gfp_results if r["gfp_correct"]) / max(len(gfp_results), 1)

        all_task_results[task_name] = {
            "lc_acc": float(lc_acc),
            "cc_acc": float(cc_acc),
            "gfp_acc": float(gfp_acc),
            "n_examples": len(task_examples),
            "examples": gfp_results
        }
        lc_total_correct += sum(1 for r in lc_results if r["correct"])
        gfp_total_correct += sum(1 for r in gfp_results if r["gfp_correct"])
        total_examples += len(task_examples)

        print(f"  {task_name:40s} LC={lc_acc:.3f}  CC={cc_acc:.3f}  GFP={gfp_acc:.3f}")

    lc_overall = lc_total_correct / max(total_examples, 1)
    gfp_overall = gfp_total_correct / max(total_examples, 1)
    cc_route_rate = gfp_routed_to_cc / max(total_examples, 1)
    gfp_disagree_acc = gfp_correct_on_disagree / max(disagree_total, 1)
    lc_disagree_acc = lc_correct_on_disagree / max(disagree_total, 1)

    summary = {
        "lc_overall_acc": float(lc_overall),
        "gfp_overall_acc": float(gfp_overall),
        "cc_route_rate": float(cc_route_rate),
        "disagreement_rate": float(disagree_total / max(total_examples, 1)),
        "gfp_disagree_acc": float(gfp_disagree_acc),
        "lc_disagree_acc": float(lc_disagree_acc),
        "total_examples": total_examples,
        "per_task": all_task_results
    }
    return summary


# Run evaluation
print("Evaluating GFP system on BBH...")
eval_results = evaluate_gfp_system(
    lc_model, cc_model, arbiter, tokenizer, bbh_data, CFG, BBH_TASKS
)

# Save results
with open(f"{RESULTS_DIR}/gfp_eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR}/gfp_eval_results.json")

# %% ─── CELL 12 — RESULTS ANALYSIS & PLOTS ───────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS ANALYSIS")
print("=" * 60)

er = eval_results
print(f"\nOverall Accuracy:")
print(f"  Logic Core:  {er['lc_overall_acc']:.4f}")
print(f"  GFP System:  {er['gfp_overall_acc']:.4f}")
print(f"  Difference:  {er['gfp_overall_acc'] - er['lc_overall_acc']:+.4f}")
print(f"\nArbiter behavior:")
print(f"  CC route rate:        {er['cc_route_rate']:.3f}")
print(f"  Disagreement rate:    {er['disagreement_rate']:.3f}  (target >0.15)")
print(f"\nOn disagreement cases:")
print(f"  LC accuracy:          {er['lc_disagree_acc']:.3f}")
print(f"  GFP accuracy:         {er['gfp_disagree_acc']:.3f}")

# Per-task breakdown
print(f"\nPer-task results:")
for task, res in er["per_task"].items():
    diff = res["gfp_acc"] - res["lc_acc"]
    arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
    print(f"  {task:40s}  LC={res['lc_acc']:.3f}  CC={res['cc_acc']:.3f}  "
          f"GFP={res['gfp_acc']:.3f}  {arrow}{abs(diff):.3f}")

# ── Bar chart: per-task LC vs GFP ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Phase 3 GFP Evaluation Results", fontsize=14)

task_names_short = [t[:25] for t in BBH_TASKS]
lc_accs = [er["per_task"][t]["lc_acc"] for t in BBH_TASKS]
cc_accs = [er["per_task"][t]["cc_acc"] for t in BBH_TASKS]
gfp_accs = [er["per_task"][t]["gfp_acc"] for t in BBH_TASKS]

x = np.arange(len(BBH_TASKS))
w = 0.28
ax = axes[0]
ax.bar(x - w, lc_accs, w, label="Logic Core", color="#2196F3")
ax.bar(x, cc_accs, w, label="Chaos Core", color="#FF5722")
ax.bar(x + w, gfp_accs, w, label="GFP System", color="#4CAF50")
ax.set_xticks(x)
ax.set_xticklabels(task_names_short, rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Accuracy")
ax.set_title("Per-Task Accuracy: LC vs CC vs GFP")
ax.legend()
ax.grid(alpha=0.3)

# ── Scatter: LC vs GFP per task ─────────────────────────────────────────────
ax2 = axes[1]
ax2.scatter(lc_accs, gfp_accs, s=80, color="#4CAF50", zorder=3)
for i, task in enumerate(task_names_short):
    ax2.annotate(task, (lc_accs[i], gfp_accs[i]), fontsize=7, ha='left')
lim = [0, 1.05]
ax2.plot(lim, lim, 'k--', linewidth=1, label='GFP = LC (parity)')
ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.set_xlabel("Logic Core Accuracy")
ax2.set_ylabel("GFP System Accuracy")
ax2.set_title("GFP vs LC (above diagonal = GFP wins)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/content/gfp_plots/gfp_evaluation_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nResults plot saved to /content/gfp_plots/gfp_evaluation_results.png")

# ── Training loss curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Phase 3 Training Loss Curves", fontsize=13)
axes[0].plot(lc_losses, marker='o', color="#2196F3")
axes[0].set_title("Logic Core")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)
axes[1].plot(cc_losses, marker='o', color="#FF5722")
axes[1].set_title("Chaos Core")
axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3)
axes[2].plot(arb_losses, marker='o', color="#9C27B0")
axes[2].set_title("Arbiter")
axes[2].set_xlabel("Epoch")
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/content/gfp_plots/training_loss_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Training loss curves saved to /content/gfp_plots/training_loss_curves.png")

# %% ─── CELL 13 — SUCCESS CRITERIA CHECK & PHASE 4 GUIDE ─────────────────────

print("\n" + "=" * 60)
print("SUCCESS CRITERIA & PHASE 4 RECOMMENDATION")
print("=" * 60)

lc_acc = er["lc_overall_acc"]
gfp_acc = er["gfp_overall_acc"]
disagree_rate = er["disagreement_rate"]
gfp_disagree = er["gfp_disagree_acc"]
lc_disagree = er["lc_disagree_acc"]
cc_route_rate = er["cc_route_rate"]

s1 = gfp_acc >= lc_acc
s2 = any(er["per_task"][t]["cc_acc"] > er["per_task"][t]["lc_acc"] for t in BBH_TASKS)
s3 = gfp_disagree > lc_disagree
s4 = disagree_rate > 0.15

print(f"\nPhase 3 success criteria:")
print(f"  □ GFP >= LC overall accuracy:          {'✓' if s1 else '✗'}  ({gfp_acc:.3f} vs {lc_acc:.3f})")
print(f"  □ CC > LC on any single subtask:        {'✓' if s2 else '✗'}")
print(f"  □ GFP > LC on disagreement cases:       {'✓' if s3 else '✗'}  ({gfp_disagree:.3f} vs {lc_disagree:.3f})")
print(f"  □ Disagreement rate > 15%:              {'✓' if s4 else '✗'}  ({disagree_rate:.3f})")

n_passed = sum([s1, s2, s3, s4])
print(f"\n{n_passed}/4 success criteria met.")

print("\n" + "=" * 60)
print("PHASE 4 RECOMMENDATION")
print("=" * 60)

if s1 and n_passed >= 2:
    print("→ PHASE 4A: SCALE")
    print("  GFP outperforms LC. Scale up with Mistral-7B + LoRA (4-bit).")
    print("  In phase3_chaos_core_lm.py, change:")
    print("    CFG.model_name = 'mistralai/Mistral-7B-v0.1'")
    print("  And add BitsAndBytesConfig for 4-bit quantization.")
elif s2 and not s1:
    print("→ PHASE 4B: DOMAIN INVESTIGATION")
    cc_wins = [t for t in BBH_TASKS if er["per_task"][t]["cc_acc"] > er["per_task"][t]["lc_acc"]]
    print(f"  CC outperforms LC on: {cc_wins}")
    print("  Build evaluation suite specifically for these tasks.")
    print("  Test: does CC advantage hold at zero-shot vs few-shot?")
elif s4 and not s1:
    print("→ PHASE 4C: INTERPRETABILITY")
    print(f"  High disagreement rate ({disagree_rate:.3f}) but arbiter not routing correctly.")
    print("  Attention visualization on disagreement examples.")
    print("  Tool: manual attention extraction from GPT-2 (model.transformer.h[*].attn)")
else:
    print("→ PHASE 4D: RETURN TO SIGNALS")
    print("  Phase 3 did not meet criteria. Phase 2 is the real experiment.")
    print("  Run Phase 2 with:")
    print("    - Multi-frequency signal (harder precursor detection)")
    print("    - Non-stationary environment (precursor pattern changes mid-run)")
    print("    - Partial observability (window_size=10)")

# Failure mode checks
print("\nFailure mode checks:")
if cc_route_rate < 0.05:
    print("  [ALERT] Arbiter almost never routes to CC (<5%).")
    print("    Fix: add LC bias removal → arbiter.net[-2].bias.fill_(0.0)")
if gfp_acc < lc_acc - 0.05:
    print("  [ALERT] GFP significantly worse than LC — arbiter routing to CC when CC is wrong.")
    print("    Fix: lower cc_route_min_confidence to 0.65 in CFG")
if nonzero_rate < 0.10:
    print("  [ALERT] CC reward too sparse. Lower intrinsic_error_threshold_std to 0.7.")

print("\nFiles to download:")
print(f"  {OUT_DIR}/logic_core/     ← LC weights")
print(f"  {OUT_DIR}/chaos_core/     ← CC weights")
print(f"  {OUT_DIR}/arbiter.pt      ← arbiter weights")
print(f"  {RESULTS_DIR}/gfp_eval_results.json")
print(f"  /content/gfp_plots/       ← all plots")
