import os
import torch
import requests
import json
import time
import random
import inspect
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset

# --- CONFIGURATION (OPTIMIZED FOR RTX 3050 6GB) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for 6GB VRAM
OUTPUT_DIR = "./soc_master_model"
SOC_ENV_URL = "http://localhost:7860"

# RTX 3050 Memory Optimizations
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
            
            # Extract JSON action
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                rewards.append(0.0)
                continue
                
            action_json = content[start:end]
            resp = requests.post(
                f"{SOC_ENV_URL}/step", 
                json=json.loads(action_json),
                timeout=5
            )
            data = resp.json()
            r = data.get("reward", {}).get("step_reward", 0.0)
            rewards.append(float(r))
        except Exception as e:
            print(f"Reward Error: {e}")
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
            prompt = f"SYSTEM: You are a SOC Analyst. Task: {task_name}. Identify compromise and remediate.\nSTATE: {json.dumps(obs['data'])}\nACTION (JSON): "
            data["prompt"].append(prompt)
            data["seed"].append(s)
        except Exception as e:
            print(f"Sampling Error: {e}")
            continue
    
    return Dataset.from_dict(data)

def train():
    print("Initializing SOC-Env Master Training (RTX 3050 Optimized)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    
    # Curriculum Training
    for tier in [1, 2]:
        print(f"\n🚀 STARTING TIER {tier} TRAINING")
        dataset = get_curriculum_dataset(tier=tier, count=50)
        
        # BULLETPROOF ARGUMENT ROUTING
        # Handles TRL version drift once and for all.
        config_params = {
            "output_dir": f"{OUTPUT_DIR}_tier{tier}",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_generations": 4,
            "max_steps": 100,
            "logging_steps": 5,
            "fp16": True,
            "report_to": "none"
        }
        
        lengths = {"max_prompt_length": 768, "max_completion_length": 128}
        
        # Check if GRPOConfig accepts the length arguments
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
        
        trainer.train()
        model.save_pretrained(f"{OUTPUT_DIR}_final_tier{tier}")

if __name__ == "__main__":
    train()
