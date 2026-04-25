import os
import torch
import requests
import json
import time
import random
import re
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# --- CONFIGURATION (STABLE CORE - NO TRL DEPENDENCIES) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
OUTPUT_DIR = "./soc_master_model"
SOC_ENV_URL = "http://localhost:7860"

# Hyperparameters
LR = 1e-5                # Slightly lower for stability
NUM_GENERATIONS = 4      # Completions per prompt (Group Size)
MAX_STEPS = 100
GRAD_ACCUM = 4
MAX_PROMPT_LEN = 1024
MAX_NEW_TOKENS = 128

# Quantization
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- CORE LOGIC: REWARD & ENVIRONMENT ---
def get_reward(prompt, completion, seed):
    """Parses JSON from completion and gets reward from environment."""
    format_bonus = 0.0
    try:
        # Robust JSON extraction
        match = re.search(r'(\{.*\})', completion, re.DOTALL)
        if match:
            action_json = match.group(1)
            try:
                # Small bonus just for being valid JSON (helps the model learn the structure)
                action_data = json.loads(action_json)
                format_bonus = 0.05 
                
                # Call environment
                requests.post(f"{SOC_ENV_URL}/reset?task=task_1&seed={seed}", timeout=5)
                resp = requests.post(f"{SOC_ENV_URL}/step", json=action_data, timeout=5)
                env_reward = float(resp.json().get("reward", {}).get("step_reward", 0.0))
                
                return env_reward + format_bonus
            except:
                pass
        return 0.0
    except Exception:
        return 0.0

# --- DATASET GENERATOR ---
def get_curriculum_data(tier=1, count=50):
    data = []
    tasks = [f"task_{i}" for i in range(1, 5)]
    task_name = tasks[tier-1]
    print(f"Sampling {count} states for {task_name}...")
    for _ in range(count):
        seed = random.randint(0, 1000000)
        try:
            resp = requests.post(f"{SOC_ENV_URL}/reset?task={task_name}&seed={seed}", timeout=5)
            obs = resp.json()
            prompt = f"SYSTEM: You are a SOC Analyst. Task: {task_name}. Identify compromise and remediate.\nSTATE: {json.dumps(obs['data'])}\nACTION (JSON): "
            data.append({"prompt": prompt, "seed": seed})
        except: continue
    return data

# --- CUSTOM GRPO TRAINING LOOP ---
def train():
    print(f"Initializing Stable GRPO Kickstart for {MODEL_ID}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Important: prepare_model_for_kbit_training enables gradient checkpointing
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    
    # KICKSTART FIX: Enable gradients for input embeddings to satisfy checkpointing
    model.enable_input_require_grads()
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=LR)
    
    for tier in [1]: # Focus on Tier 1 for the kickstart
        print(f"\n🚀 STARTING TIER {tier} TRAINING")
        dataset = get_curriculum_data(tier=tier, count=50)
        
        pbar = tqdm(range(MAX_STEPS))
        step = 0
        optimizer.zero_grad()
        
        for i in pbar:
            sample = random.choice(dataset)
            prompt = sample["prompt"]
            seed = sample["seed"]
            
            # 1. GENERATE GROUP
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT_LEN).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_return_sequences=NUM_GENERATIONS,
                    do_sample=True,
                    temperature=0.9, # Higher temp to encourage exploration
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # 2. EVALUATE REWARDS
            completions = [tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True) for out in outputs]
            rewards = torch.tensor([get_reward(prompt, c, seed) for c in completions], device="cuda", dtype=torch.float32)
            
            # 3. COMPUTE GRPO ADVANTAGES
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8
            advantages = (rewards - mean_r) / std_r
            
            # 4. COMPUTE LOG PROBS & LOSS
            all_tokens = outputs # [GroupSize, FullSeqLen]
            prompt_len = inputs.input_ids.shape[1]
            
            # Re-run forward pass on generated outputs to get differentiable logits
            # Use input_ids directly (model.enable_input_require_grads() handles the checkpointing requirement)
            res = model(all_tokens)
            logits = res.logits[:, prompt_len-1:-1, :] 
            labels = all_tokens[:, prompt_len:].contiguous()
            
            log_probs = torch.log_softmax(logits, dim=-1)
            target_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
            
            mask = (labels != tokenizer.pad_token_id).float()
            per_token_log_probs = target_log_probs * mask
            sentence_log_probs = per_token_log_probs.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
            loss = -(sentence_log_probs * advantages).mean()
            loss = loss / GRAD_ACCUM
            
            if loss != 0:
                loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            pbar.set_description(f"Loss: {loss.item()*GRAD_ACCUM:.4f} | R_mean: {mean_r:.3f}")

        model.save_pretrained(f"{OUTPUT_DIR}_final_tier{tier}")
    
    print("Training Complete!")

if __name__ == "__main__":
    train()
