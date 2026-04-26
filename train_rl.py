"""
🚀 SOC-Env: Native Master Trainer (Zero-Dependency GRPO)
—————————————————————————————————————————————————————
• Scrap Blackboxes: No trl.GRPOTrainer, No Unsloth.
• 100% Stable: Uses standard transformers + peft + bitsandbytes.
• OpenEnv: Inherits and uses the native environment logic.
• VRAM Optimized: Fits in 6GB (Laptop) and T4 (Colab).
• Features: Curriculum, Multi-signal Reward, Real-time Graphs.
"""

import os, json, random, re, time, warnings
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from soc_openenv import make_env

warnings.filterwarnings("ignore")

# ── CONFIGURATION ────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = "./soc_rl_stable_final"
GRAPH_DIR  = "./training_graphs"
MAX_STEPS  = 120   # total optimizer steps
NUM_GENS   = 4     # responses per state (GRPO group size)
LR         = 2e-5
os.makedirs(GRAPH_DIR, exist_ok=True)

# ── REWARD LOGIC (OpenEnv Managed) ───────────────────────────────────────
def calculate_reward(prompt, response, task_id):
    """Zero-dependency reward calculator."""
    score = 0.0
    # 1. Format Signal (Fast)
    if "{" in response and "}" in response: score += 0.1
    match = re.search(r"(\{.*\})", response, re.DOTALL)
    if not match: return score

    try:
        action_data = json.loads(match.group(1))
        score += 0.2
        # 2. OpenEnv Signal (Depth)
        env = make_env(task_id=task_id)
        env.reset()
        _, env_rew, _, _ = env.step(action_data)
        # We value the environment's score highly
        score += (float(env_rew) * 1.0)
    except:
        pass
    return min(1.5, score)

# ── TRAINING LOOP ────────────────────────────────────────────────────────
def train():
    print(f"🎬 Starting Zero-Blackbox Training: {MODEL_ID}")
    
    # 1. Load Model (Stable 4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=1e-6)

    # TensorBoard & Logging
    tb_writer = SummaryWriter(log_dir=os.path.join(GRAPH_DIR, "tensorboard"))
    history = []
    pbar = tqdm(range(MAX_STEPS), desc="Optimizing")

    for step in pbar:
        # 1. Curriculum Task Selection
        if step < 20:   task_id = "task_1"
        elif step < 50: task_id = random.choice(["task_1", "task_2"])
        elif step < 80: task_id = random.choice(["task_1", "task_2", "task_3"])
        else:           task_id = random.choice(["task_1", "task_2", "task_3", "task_4"])

        # 2. Get Env State
        env = make_env(task_id=task_id)
        obs = env.reset()
        state_str = json.dumps(json.loads(obs.model_dump_json())["data"])
        prompt = f"SYSTEM: SOC Analyst. Reply JSON.\\nSTATE: {state_str}\\nACTION:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 3. GRPO Generation (4 completions)
        completions = []
        rewards = []
        
        with torch.no_grad():
            for _ in range(NUM_GENS):
                out = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
                resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                completions.append(out)
                rewards.append(calculate_reward(prompt, resp, task_id))

        # 4. Compute Advantages
        r_mean = sum(rewards) / len(rewards)
        r_std  = (sum((r - r_mean)**2 for r in rewards) / len(rewards))**0.5 + 1e-6
        advantages = [(r - r_mean) / r_std for r in rewards]

        # 5. Optimization Step
        total_loss = 0
        optimizer.zero_grad()
        
        for i in range(NUM_GENS):
            outputs = model(completions[i])
            logits  = outputs.logits[:, inputs.input_ids.shape[1]-1 : -1, :]
            labels  = completions[i][:, inputs.input_ids.shape[1] : ]
            
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            loss = -(target_log_probs.mean() * advantages[i])
            
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        # ── Requirements: Log to TensorBoard & History ──
        mean_r = sum(rewards) / len(rewards)
        history.append(mean_r)
        
        opt_loss = total_loss / NUM_GENS
        tb_writer.add_scalar("Reward/Mean", mean_r, step)
        tb_writer.add_scalar("Loss/Policy", opt_loss, step)
        tb_writer.add_scalar("Curriculum/TaskIndex", ["task_1","task_2","task_3","task_4"].index(task_id), step)

        pbar.set_postfix({"R": f"{mean_r:.2f}", "Loss": f"{opt_loss:.4f}", "Task": task_id})

        # ── Requirements: Real-time Matplotlib Graph ──
        if step > 0 and step % 20 == 0:
            plt.figure(figsize=(10,5))
            plt.plot(history, color="#2563eb", linewidth=2, label="Agent Reward")
            plt.axhline(0.35, color="#dc2626", linestyle="--", alpha=0.6, label="Base Format Score")
            plt.fill_between(range(len(history)), history, 0.35, where=(torch.tensor(history) > 0.35), color="#dbeafe", alpha=0.5)
            plt.title(f"SOC-Env Progress: Step {step} ({task_id})", fontsize=12)
            plt.xlabel("Optimizer Steps", fontsize=10)
            plt.ylabel("Mean Episode Reward", fontsize=10)
            plt.legend()
            plt.grid(alpha=0.2)
            plt.savefig(f"{GRAPH_DIR}/reward_curve.png", dpi=120)
            plt.close()

    # Final Save
    tb_writer.close()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
