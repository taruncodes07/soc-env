"""
SOC-Env: High-Performance Custom GRPO RL Training
—————————————————————————————————————————————
• NO trl.GRPOTrainer (fragile, breaks constantly)
• NO HTTP server dependency (environment called natively)
• Curriculum: Task 1 → Task 4 with progressive unlocking
• Rich multi-signal reward: format + environment step + episode final
• Reward curve PNG saved every checkpoint
• RTX 3050 (6GB) safe: FP16, 4-bit quant, sequential micro-batches
"""

import os, json, random, re, time, warnings
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works on Colab/headless)
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Direct native environment imports — bypasses HTTP server entirely
from server.environment import Environment
from server.models import Action

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*BitsAndBytes.*")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
# Qwen2.5-3B is 2x smarter than 1.5B, still fits in 6GB at 4-bit.
# On Colab T4/A100 you can bump this to "Qwen/Qwen2.5-7B-Instruct".
MODEL_ID   = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./soc_rl_trained"
GRAPH_DIR  = "./training_graphs"

LR          = 2e-5     # Higher LR: format already learned from Phase 1
GROUP_SIZE  = 4        # 4 completions per prompt — safe for 6GB VRAM
GRAD_ACCUM  = 4        # Effective batch = GROUP_SIZE * GRAD_ACCUM = 16
MAX_STEPS   = 150      # 150 optimizer steps → 600 inner generations
SAVE_EVERY  = 50       # Save model + graph every N optimizer steps
MAX_PROMPT_LEN = 700
MAX_NEW_TOKENS = 180

# Curriculum progression (tasks unlocked as steps advance)
CURRICULUM = {
    0:   ["task_1"],
    30:  ["task_1", "task_2"],
    70:  ["task_1", "task_2", "task_3"],
    110: ["task_1", "task_2", "task_3", "task_4"],
}

# ─────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ─────────────────────────────────────────────────────────────────
# NATIVE ENVIRONMENT HELPERS
# ─────────────────────────────────────────────────────────────────
def run_full_episode_reward(action_data: dict, task_id: str) -> float:
    """
    Runs a short, guided episode to get a rich reward signal.
    First step = model's action. Remaining steps = greedy optimal play
    so the episode actually terminates and produces a final_score.
    Returns a reward in [0, 1].
    """
    env = Environment()
    try:
        env.reset(task_id=task_id, randomize=True)
    except Exception:
        return 0.0

    # Step 1: Execute the model's proposed action
    try:
        action_obj = Action(**action_data)
        _, reward_obj, done, _ = env.step(action_obj)
        step_reward = float(reward_obj.step_reward)
    except Exception:
        step_reward = -0.05
        done = False

    if done:
        return max(0.0, min(1.0, step_reward))

    # Steps 2+: Greedy oracle play to drive the episode to completion
    # and produce a meaningful final_score for RL shaping
    gt = env.state.ground_truth
    max_oracle = env.state.max_steps - env.state.step_number

    for _ in range(max_oracle):
        if env.state.done:
            break
        # Oracle strategy: investigate → remediate → escalate → mark_safe
        acted = False
        for dev_id in gt.compromised_devices:
            if dev_id not in env.state.investigated_devices:
                try:
                    env.step(Action(action="investigate_device", target_device=dev_id))
                    acted = True; break
                except Exception: pass

        if not acted:
            for dev_id in gt.compromised_devices:
                dev_state = env.state.devices.get(dev_id)
                if dev_state and not dev_state.is_isolated:
                    # Pick correct remediation action
                    req = gt.correct_remediation_actions.get(dev_id, ["isolate_device"])
                    for r_action in req:
                        try:
                            if r_action == "kill_process":
                                proc = (dev_state.telemetry.active_processes or ["unknown_proc"])[-1]
                                env.step(Action(action="kill_process", target_device=dev_id, target_process=proc))
                            elif r_action == "block_ip":
                                ips = dev_state.telemetry.network.flagged_ips
                                ip = ips[0] if ips else "0.0.0.0"
                                env.step(Action(action="block_ip", target_device=dev_id, target_ip=ip))
                            else:
                                env.step(Action(action=r_action, target_device=dev_id))
                            acted = True; break
                        except Exception: pass
                    if acted: break

        if not acted:
            for dev_id in gt.compromised_devices:
                if dev_id not in env.state.escalated_devices:
                    a_type = gt.anomaly_types.get(dev_id, "malware_process")
                    try:
                        env.step(Action(action="escalate", target_device=dev_id, anomaly_type=a_type))
                        acted = True; break
                    except Exception: pass

        if not acted:
            for dev_id in gt.healthy_devices:
                if dev_id not in env.state.marked_safe_devices:
                    try:
                        env.step(Action(action="mark_safe", target_device=dev_id))
                        acted = True; break
                    except Exception: pass

        if not acted:
            try: env.step(Action(action="poll_org"))
            except Exception: break

    # Get final episode score
    final = env.state.final_score
    if final is None:
        final = env.state.identification_score + env.state.remediation_score
    final = max(0.0, min(1.0, float(final)))

    # Blend: 30% model's first-step reward, 70% episode outcome
    return 0.3 * max(0.0, step_reward) + 0.7 * final


def get_reward(completion: str, task_id: str) -> float:
    """
    Multi-signal reward function:
    1. Format signal  (fast, always runs)
    2. Environment signal (native, no HTTP)
    Returns a float reward.
    """
    score = 0.0

    # Signal 1: Attempted structure
    if "action" in completion.lower() or "{" in completion:
        score += 0.05

    if "{" in completion and "}" in completion:
        score += 0.10

    # Signal 2: Valid JSON extraction
    match = re.search(r'(\{[^{}]*\})', completion, re.DOTALL)
    if not match:
        return score

    try:
        action_data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return score

    score += 0.20  # Valid JSON bonus

    # Signal 3: Has valid action key
    valid_actions = {
        "poll_org", "investigate_device", "isolate_device",
        "block_ip", "kill_process", "escalate", "mark_safe"
    }
    if action_data.get("action") in valid_actions:
        score += 0.10  # Knows valid action names

    # Signal 4: Has target device (shows strategic thinking)
    if action_data.get("action") != "poll_org" and action_data.get("target_device"):
        score += 0.10

    # Signal 5: Full native environment episode score
    try:
        env_score = run_full_episode_reward(action_data, task_id)
        score += env_score * 0.80  # Environment is the primary driver
    except Exception:
        pass

    return min(score, 1.5)  # Uncapped to allow high rewards to propagate


# ─────────────────────────────────────────────────────────────────
# GRAPH SAVING
# ─────────────────────────────────────────────────────────────────
def save_reward_graph(steps_log, avg_reward_log, max_reward_log, loss_log, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(steps_log, avg_reward_log, label="Avg Reward", color="#4A9EFF", linewidth=2)
    axes[0].plot(steps_log, max_reward_log, label="Max Reward", color="#50C878",
                 linewidth=1.5, linestyle="--", alpha=0.8)
    axes[0].axhline(y=0.35, color="#FF6B6B", linestyle=":", linewidth=1.5,
                    label="Phase 1 Ceiling (Format Only)")
    axes[0].set_title("SOC-Env Agent: Reward Progress", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Reward")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.6)

    if loss_log:
        axes[1].plot(steps_log, loss_log, label="GRPO Loss", color="#FF8C42", linewidth=2)
        axes[1].set_title("Training Loss", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Optimizer Step")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 Graph saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────
def get_active_tasks(optimizer_step: int) -> list:
    """Returns curriculum task list for current step."""
    active = ["task_1"]
    for threshold, tasks in CURRICULUM.items():
        if optimizer_step >= threshold:
            active = tasks
    return active


def train():
    os.makedirs(GRAPH_DIR, exist_ok=True)

    print(f"🚀  Loading {MODEL_ID} in 4-bit FP16 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token  = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = get_peft_model(model, LORA_CONFIG)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, eps=1e-8, weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=1e-6)

    # Pre-sample starting states for all tasks natively
    print("📂  Sampling initial states from all tasks ...")
    training_pool = {task: [] for task in ["task_1", "task_2", "task_3", "task_4"]}
    sample_env = Environment()
    for task_id in training_pool:
        for _ in range(25):
            try:
                obs = sample_env.reset(task_id=task_id, randomize=True)
                obs_dict = json.loads(obs.model_dump_json())
                training_pool[task_id].append({
                    "state": json.dumps(obs_dict["data"]),
                    "task_id": task_id
                })
            except Exception:
                continue

    total_tasks = sum(len(v) for v in training_pool.values())
    print(f"✅  Loaded {total_tasks} states across {len(training_pool)} tasks.\n")

    # Tracking logs for graphs
    steps_log, avg_reward_log, max_reward_log, loss_log = [], [], [], []

    inner_step  = 0
    opt_step    = 0
    model.train()
    optimizer.zero_grad()

    pbar = tqdm(total=MAX_STEPS, desc="Training", unit="opt-step")

    while opt_step < MAX_STEPS:
        # ── Curriculum: pick from currently unlocked tasks ──
        active_tasks = get_active_tasks(opt_step)
        task_id = random.choice(active_tasks)
        pool    = training_pool.get(task_id, training_pool["task_1"])
        if not pool:
            continue
        sample  = random.choice(pool)

        prompt = (
            "SYSTEM: You are a professional SOC Analyst. Given the current network state, "
            "output exactly ONE action as a JSON object. "
            "Valid actions: poll_org, investigate_device, isolate_device, block_ip, kill_process, escalate, mark_safe. "
            "For device actions always include target_device. "
            "Prioritise CRITICAL and high-criticality devices that show multiple alert flags.\n"
            f"CURRENT STATE:\n{sample['state']}\n"
            "RESPOND WITH JSON ONLY:\n"
        )

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=MAX_PROMPT_LEN, padding=False
        ).to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # ── GENERATE (no grad) ──
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=GROUP_SIZE,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()
        model.train()

        # ── REWARD ──
        completions  = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in outputs
        ]
        raw_rewards  = [get_reward(c, task_id) for c in completions]
        rewards_t    = torch.tensor(raw_rewards, device="cuda", dtype=torch.float32)

        mean_r = rewards_t.mean().item()
        max_r  = rewards_t.max().item()

        # ── ADVANTAGES ──
        std_r = rewards_t.std().item()
        if std_r < 1e-4:
            # Degenerate case: all same reward — use sign of mean to nudge
            advantages = torch.full_like(rewards_t, 0.1 if mean_r > 0.1 else -0.1)
        else:
            advantages = (rewards_t - rewards_t.mean()) / (std_r + 1e-8)

        # ── TRAINING UPDATE (sequential micro-batches) ──
        total_loss = 0.0
        for i in range(GROUP_SIZE):
            single_out = outputs[i:i+1]           # (1, seq_len)
            adv_i      = advantages[i].float()

            # Cast logits to float32 for stable log_softmax
            logits = model(single_out).logits[:, prompt_len-1:-1, :].float()
            labels = single_out[:, prompt_len:].contiguous()

            log_probs          = torch.log_softmax(logits, dim=-1)
            per_tok_lp         = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
            mask               = (labels != tokenizer.pad_token_id).float()
            response_len       = mask.sum(dim=1) + 1e-8
            mean_log_prob      = (per_tok_lp * mask).sum(dim=1) / response_len
            # GRPO loss: negative mean log-prob scaled by advantage
            single_loss        = -(mean_log_prob * adv_i).mean() / GRAD_ACCUM

            if not torch.isnan(single_loss) and single_loss.requires_grad:
                single_loss.backward()
                total_loss += single_loss.item()

            del logits, log_probs, per_tok_lp, labels, mask, mean_log_prob, single_loss
            torch.cuda.empty_cache()

        inner_step += 1

        # ── OPTIMIZER STEP ──
        if inner_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

            # Log
            steps_log.append(opt_step)
            avg_reward_log.append(mean_r)
            max_reward_log.append(max_r)
            loss_log.append(total_loss)

            # Update tqdm bar
            cur_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "task": task_id,
                "R_avg": f"{mean_r:.3f}",
                "R_max": f"{max_r:.3f}",
                "loss": f"{total_loss:.4f}",
                "lr": f"{cur_lr:.2e}",
            })
            pbar.update(1)

            # ── CHECKPOINT ──
            if opt_step % SAVE_EVERY == 0 or opt_step == MAX_STEPS:
                ckpt = f"{OUTPUT_DIR}_step{opt_step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"\n💾  Checkpoint saved → {ckpt}")

                graph_path = os.path.join(GRAPH_DIR, f"reward_curve_step{opt_step}.png")
                save_reward_graph(
                    steps_log, avg_reward_log, max_reward_log, loss_log, graph_path
                )

    pbar.close()

    # ── FINAL SAVE ──
    model.save_pretrained(OUTPUT_DIR + "_final")
    tokenizer.save_pretrained(OUTPUT_DIR + "_final")

    # Final graph
    final_graph = os.path.join(GRAPH_DIR, "reward_curve_FINAL.png")
    save_reward_graph(steps_log, avg_reward_log, max_reward_log, loss_log, final_graph)

    print(f"\n✅  Training complete!")
    print(f"   Model  → {OUTPUT_DIR}_final")
    print(f"   Graphs → {GRAPH_DIR}/")


if __name__ == "__main__":
    train()
