import os
import torch
import requests
import json
import time
import random
import inspect
import re
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset

# --- CONFIGURATION (OPTIMIZED FOR RTX 3050 6GB/COLAB T4) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
OUTPUT_DIR = "./soc_master_model"
SOC_ENV_URL = "http://localhost:7860"

# RTX 3050 Memory Optimizations
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # Force Float16 to avoid BF16 AMP issues
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

# --- REWARD FUNCTIONS ---
def soc_reward_func(prompts, completions, seed, **kwargs):
    """
    Calls the SOC-Env server with a deterministic seed to evaluate completions.
    """
    rewards = []
    for i, content in enumerate(completions):
        try:
            current_seed = seed[i]
            requests.post(f"{SOC_ENV_URL}/reset?task=task_1&seed={current_seed}")
            
            # IMPROVED JSON EXTRACTION: Find the outermost valid brace pair
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue
                
            action_json = match.group(1)
            # Handle potential extra characters if the greedy match captures too much
            try:
                action_data = json.loads(action_json)
            except json.JSONDecodeError:
                # Fallback: try to find the start and end manually
                start = action_json.find('{')
                end = action_json.rfind('}') + 1
                action_data = json.loads(action_json[start:end])

            resp = requests.post(
                f"{SOC_ENV_URL}/step", 
                json=action_data,
                timeout=5
            )
            data = resp.json()
            r = data.get("reward", {}).get("step_reward", 0.0)
            rewards.append(float(r))
        except Exception as e:
            # Silently handle parsing errors to keep training moving
            rewards.append(0.0)
    return rewards

# --- CURRICULUM DATA GENERATOR ---
def get_curriculum_dataset(tier=1, count=50):
    """
    Generates a dataset of (prompt, seed) pairs.
    """
    data = {"prompt": [], "seed": []}
    tasks = ["task_1", "task_2", "task_3", "task_4"]
    task_name = tasks[tier-1]
    
    print(f"Sampling {count} states for {task_name}...")
    for _ in range(count):
        s = random.randint(0, 1000000)
        try:
            resp = requests.post(f"{SOC_ENV_URL}/reset?task={task_name}&seed={s}", timeout=5)
            obs = resp.json()
            # Explicit formatting to help the model learn the structure
            prompt = f"SYSTEM: You are a SOC Analyst. Task: {task_name}. Identify compromise and remediate.\nSTATE: {json.dumps(obs['data'])}\nACTION (JSON): "
            data["prompt"].append(prompt)
            data["seed"].append(s)
        except Exception as e:
            continue
    
    return Dataset.from_dict(data)

def train():
    print("Initializing SOC-Env Master Training (Mixed Precision Optimized)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16 # Ensure weights stay out of BF16
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    
    # Curriculum Training
    for tier in [1, 2]:
        print(f"\n🚀 STARTING TIER {tier} TRAINING")
        dataset = get_curriculum_dataset(tier=tier, count=50)
        
        # BULLETPROOF CONFIG
        config_params = {
            "output_dir": f"{OUTPUT_DIR}_tier{tier}",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_generations": 4,
            "max_steps": 100,
            "logging_steps": 5,
            "fp16": True,
            "bf16": False, # EXPLICITLY DISABLE BF16 to avoid AMP GradScaler issues
            "report_to": "none"
        }
        
        lengths = {"max_prompt_length": 1024, "max_completion_length": 128}
        
        config_sig = inspect.signature(GRPOConfig.__init__).parameters
        trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters
        
        final_config_args = config_params.copy()
        final_trainer_args = {
            "model": model,
            "reward_funcs": [soc_reward_func],
            "train_dataset": dataset,
        }
        
        for k, v in lengths.items():
            if k in config_sig:
                final_config_args[k] = v
            elif k in trainer_sig:
                final_trainer_args[k] = v
        
        training_args = GRPOConfig(**final_config_args)
        trainer = GRPOTrainer(**final_trainer_args, args=training_args)
        
        # Disable automatic BF16 detection if enabled by default in some versions
        if hasattr(trainer.args, 'bf16'):
            trainer.args.bf16 = False
        
        trainer.train()
        model.save_pretrained(f"{OUTPUT_DIR}_final_tier{tier}")

if __name__ == "__main__":
    train()
