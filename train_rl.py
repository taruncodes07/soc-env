import os
import torch
import json
import time
import random
import re
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Direct Environment Imports
from server.environment import Environment
from server.models import Action

# --- STABLE FRAMEWORK (CUSTOM GRPO, NO HTTP SERVER) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
OUTPUT_DIR = "./soc_master_model"

# Optimized for 6GB VRAM (No Checkpointing needed for 1.5B 4-bit)
LR = 5e-6
GROUP_SIZE = 4       # Keep small to avoid OOM, but enough for advantage estimation
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
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

def evaluate_action_on_env(seed, action_data):
    """Initializes a local environment instance natively, safely avoids HTTP overhead."""
    env = Environment()
    # Reset natively
    try:
        obs = env.reset(task_id="task_1", randomize=True)
    except Exception as e:
        return 0.0
    
    # Try step natively
    try:
        # Create an action model, allowing missing ones to just fail elegantly
        action_obj = Action(**action_data)
        next_obs, reward_obj, done, info = env.step(action_obj)
        return float(reward_obj.step_reward)
    except Exception as e:
        return -0.02 # Small penalty for invalid action schema

def get_reward(completion, seed):
    """Calculates reward with granular structure feedback natively."""
    try:
        score = 0.0
        
        # Step 1: Give a small signal if it even tries to output an action
        if "action" in completion.lower() or "{" in completion:
            score += 0.05
            
        # Step 2: Format Heuristics
        if "{" in completion and "}" in completion:
            score += 0.1
        
        # Step 3: Extraction
        match = re.search(r'(\{.*?\})', completion, re.DOTALL)
        if not match: return score
        
        action_json = match.group(1).strip()
        try:
            action_data = json.loads(action_json)
            score += 0.2 # Valid JSON bonus
            
            # Step 4: Environment Feedback
            # Call the native simulation instead of hitting a server
            env_reward = evaluate_action_on_env(seed, action_data)
            score += env_reward * 2.0 # Scale up environment signal
            
            return score
        except json.JSONDecodeError:
            return score
    except Exception as e:
        return 0.0

def train():
    print(f"🚀 Deploying Native One-Shot RL System for {MODEL_ID}")
    print("Initializing Model and Tokenizer (Allocating VRAM...)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    model = get_peft_model(model, LORA_CONFIG)
    
    # Ensure gradients flow
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
    model.train()

    # Native State Sampling
    print("Sampling starting states natively...")
    training_data = []
    
    # We create a dummy env just to grab initial states to act as prompts
    sample_env = Environment()
    for _ in range(30):
        s = random.randint(0, 100000)
        try:
            obs = sample_env.reset(task_id="task_1", randomize=True)
            obs_dict = json.loads(obs.model_dump_json()) if hasattr(obs, 'model_dump_json') else obs.dict()
            training_data.append({"state": json.dumps(obs_dict["data"]), "seed": s})
        except Exception as e:
            continue
            
    if not training_data:
        raise RuntimeError("Failed to sample initial states from the native environment.")

    print(f"Successfully loaded {len(training_data)} states. Starting GRPO Loop...")
    pbar = tqdm(range(MAX_STEPS * GRAD_ACCUM))
    optimizer.zero_grad()
    
    for step in pbar:
        sample = random.choice(training_data)
        seed = sample["seed"]
        prompt = f"SYSTEM: You are a SOC Analyst. Analyze the state and decide on ONE action. Use JSON format. Example: {{\"action\": \"poll_org\"}}\nSTATE: {sample['state']}\nACTION: "
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LEN).to("cuda")
        
        # 1. GENERATE GROUP (No Gradients)
        model.eval() # Good practice to turn off dropout during generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=GROUP_SIZE,
                do_sample=True,
                temperature=0.9, # Needs to be high enough for variance
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        torch.cuda.empty_cache()
        model.train()
        
        # 2. EVALUATE REWARDS natively
        completions = [tokenizer.decode(o[inputs.input_ids.shape[1]:], skip_special_tokens=True) for o in outputs]
        raw_rewards = [get_reward(c, seed) for c in completions]
        rewards = torch.tensor(raw_rewards, device="cuda", dtype=torch.float16)
        
        # 3. ADVANTAGES
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        
        # CRITICAL FIX: If standard deviation is effectively zero (all models output exact same reward)
        # We enforce a small advantage or disadvantage based on if the mean score is > 0
        if rewards.std() < 1e-4:
             if mean_r > 0.05: 
                 advantages = torch.full_like(rewards, 0.1) # baseline slight push
             else:
                 advantages = torch.full_like(rewards, -0.1)
        else:
             advantages = (rewards - mean_r) / std_r
        
        # 4. TRAINING UPDATE (Gradients ON)
        # Sequential processing to guarantee 6GB VRAM safety
        total_loss_for_logging = 0.0
        
        for i in range(GROUP_SIZE):
            single_output = outputs[i:i+1] # Shape: (1, seq_len)
            single_adv = advantages[i]
            
            logits = model(single_output).logits[:, inputs.input_ids.shape[1]-1:-1, :]
            labels = single_output[:, inputs.input_ids.shape[1]:].contiguous()
            
            log_probs = torch.log_softmax(logits, dim=-1)
            per_token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
            
            mask = (labels != tokenizer.pad_token_id).float()
            # PPO surrogate proxy (we use cross-entropy scaled by advantage)
            loss_vector = -(per_token_log_probs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
            single_loss = (loss_vector * single_adv).sum() / (GROUP_SIZE * GRAD_ACCUM)
            
            if not torch.isnan(single_loss) and single_loss.requires_grad:
                single_loss.backward()
                total_loss_for_logging += single_loss.item()
                
            # Drop graph to limit VRAM scaling
            del logits, log_probs, per_token_log_probs, labels, mask, loss_vector, single_loss
            torch.cuda.empty_cache() 
        
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        pbar.set_description(f"Loss: {total_loss_for_logging:.4f} | R_Avg: {mean_r:.2f} | R_Max: {max(raw_rewards):.2f}")

    model.save_pretrained(OUTPUT_DIR + "_one_shot")
    tokenizer.save_pretrained(OUTPUT_DIR + "_one_shot")
    print("\n✅ Training Complete. Model saved natively.")

if __name__ == "__main__":
    train()
