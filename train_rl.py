import os
import torch
import requests
import json
import time
import random
import re
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- STABLE FRAMEWORK (ONE-SHOT SOLUTION) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
SOC_ENV_URL = "http://localhost:7860"
OUTPUT_DIR = "./soc_master_model"

# Optimized for 6GB VRAM (No Checkpointing needed for 1.5B 4-bit)
LR = 5e-6 # Highly stable LR
GROUP_SIZE = 8 # More samples for better advantage estimation
GRAD_ACCUM = 4
MAX_PROMPT_LEN = 800
MAX_NEW_TOKENS = 150
MAX_STEPS = 100

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=16, # Slightly higher rank for better learning
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

def get_reward(completion, seed):
    """Calculates reward with granular structure feedback."""
    try:
        score = 0.0
        
        # Give a small signal if it even tries to output an action (jumpstart gradients)
        if "action" in completion.lower() or "{" in completion:
            score += 0.05
            
        # Step 1: Format Heuristics (Quick signal)
        if "{" in completion and "}" in completion:
            score += 0.1 # General JSON-like structure
        
        # Step 2: Extraction
        match = re.search(r'(\{.*\})', completion, re.DOTALL)
        if not match: return score
        
        action_json = match.group(1).strip()
        try:
            action_data = json.loads(action_json)
            score += 0.2 # Valid JSON bonus
            
            # Step 3: Environment Feedback
            requests.post(f"{SOC_ENV_URL}/reset?task=task_1&seed={seed}", timeout=2)
            resp = requests.post(f"{SOC_ENV_URL}/step", json=action_data, timeout=2)
            env_data = resp.json()
            
            # Real task reward
            env_reward = float(env_data.get("reward", {}).get("step_reward", 0.0))
            score += env_reward * 2.0 # Scale up environment signal
            
            return score
        except:
            return score
    except:
        return 0.0

def train():
    print(f"🚀 Deploying One-Shot RL System for {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # We DO NOT use prepare_model_for_kbit_training to avoid forced checkpointing
    # instead we manually set the bits for LoRA compatibility
    model = get_peft_model(model, LORA_CONFIG)
    
    # Ensure gradients flow
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
    model.train()

    # Curriculum state Sampling
    print("Sampling tasks...")
    training_data = []
    for _ in range(30):
        s = random.randint(0, 100000)
        try:
            r = requests.post(f"{SOC_ENV_URL}/reset?task=task_1&seed={s}", timeout=2).json()
            if 'data' in r:
                training_data.append({"state": json.dumps(r['data']), "seed": s})
        except Exception as e:
            continue
            
    if not training_data:
        raise RuntimeError(f"Error: Empty dataset. Could not sample tasks from {SOC_ENV_URL}. Note: If you are running on Google Colab, make sure you start the environment server FIRST using `!python -m uvicorn server.app:app --host 127.0.0.1 --port 7860 &` before running this script.")

    pbar = tqdm(range(MAX_STEPS * GRAD_ACCUM))
    optimizer.zero_grad()
    
    for step in pbar:
        sample = random.choice(training_data)
        seed = sample["seed"]
        prompt = f"SYSTEM: Use JSON to identify compromise and remediate. Example: {{\"action\": \"poll_org\"}}\nSTATE: {sample['state']}\nACTION: "
        
        # 1. GENERATE GROUP (No Gradients)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LEN).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=GROUP_SIZE,
                do_sample=True,
                temperature=1.0, # High exploration
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 2. EVALUATE REWARDS
        # Full tokens for backprop (outputs)
        completions = [tokenizer.decode(o[inputs.input_ids.shape[1]:], skip_special_tokens=True) for o in outputs]
        raw_rewards = [get_reward(c, seed) for c in completions]
        rewards = torch.tensor(raw_rewards, device="cuda", dtype=torch.float16)
        
        # 3. GRPO ADVANTAGES
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r
        
        # 4. TRAINING UPDATE (Gradients ON)
        # We process the group sequentially (mini-batch size 1) to avoid CUDA OOM 
        # on consumer GPUs like Colab T4 (15GB) or RTX 3050 (6GB).
        has_gradients = advantages.abs().sum() > 0
        
        if has_gradients:
            for i in range(GROUP_SIZE):
                single_output = outputs[i:i+1] # Shape: (1, seq_len)
                single_adv = advantages[i]
                
                logits = model(single_output).logits[:, inputs.input_ids.shape[1]-1:-1, :]
                labels = single_output[:, inputs.input_ids.shape[1]:].contiguous()
                
                log_probs = torch.log_softmax(logits, dim=-1)
                per_token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
                
                mask = (labels != tokenizer.pad_token_id).float()
                loss_vector = -(per_token_log_probs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                
                # We calculate the mean over the group size incrementally
                single_loss = (loss_vector * single_adv).sum() / (GROUP_SIZE * GRAD_ACCUM)
                
                if not torch.isnan(single_loss):
                    single_loss.backward()
                    
                # Explicitly free intermediate tensors to keep VRAM usage perfectly flat
                del logits, log_probs, per_token_log_probs, labels, mask, loss_vector, single_loss
        
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        pbar.set_description(f"R_Avg: {mean_r:.2f} | R_Max: {max(raw_rewards):.2f} | Grad: {'Yes' if advantages.abs().sum() > 0 else 'No'}")

    model.save_pretrained(OUTPUT_DIR + "_one_shot")
    print("\n✅ Training Complete. Model saved.")

if __name__ == "__main__":
    train()
