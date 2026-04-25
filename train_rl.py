import os
import torch
import requests
import json
import time
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset

# --- CONFIGURATION (OPTIMIZED FOR RTX 3050 6GB) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" # Using a smaller model to ensure group stability on 6GB
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
def soc_reward_func(prompts, completions, **kwargs):
    """
    Calls the SOC-Env server to get rewards for the generated actions.
    """
    rewards = []
    # We strip the thought process if the model uses <thought>...
    for i, content in enumerate(completions):
        try:
            # We assume a single turn for GRPO rollout
            # Reset env for this specific sample (using sample index as seed if needed)
            requests.post(f"{SOC_ENV_URL}/reset?task=task_1")
            
            # Try to parse JSON from the completion
            # Simple heuristic: find the first { and last }
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
            # Reward is based on step_reward + meta_reward logic in server
            r = data.get("reward", {}).get("step_reward", 0.0)
            rewards.append(float(r))
        except Exception:
            rewards.append(0.0)
    return rewards

# --- CURRICULUM DATA GENERATOR ---
def get_curriculum_prompts(tier=1, count=100):
    """
    Generates a list of prompt strings from the randomized environment.
    """
    prompts = []
    tasks = ["task_1", "task_2", "task_3", "task_4"]
    task_name = tasks[tier-1]
    
    for _ in range(count):
        resp = requests.post(f"{SOC_ENV_URL}/reset?task={task_name}")
        obs = resp.json()
        # Construct the minimal prompt for the model
        prompt = f"INSTRUCTIONS: Identify compromise and remediate. State: {json.dumps(obs['data'])}"
        prompts.append(prompt)
    return prompts

def train():
    print("Initializing Master Model Training for RTX 3050...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    
    # Curriculum Loop
    for tier in [1, 2, 3, 4]:
        print(f"\n--- STARTING TIER {tier} TRAINING ---")
        
        prompt_list = get_curriculum_prompts(tier=tier, count=50) # Reduced count for faster "mastery" testing
        dataset = Dataset.from_dict({"prompt": prompt_list})
        
        training_args = GRPOConfig(
            output_dir=f"{OUTPUT_DIR}_tier{tier}",
            learning_rate=5e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_prompt_length=512,
            max_completion_length=128,
            num_generations=4, # Group size 4 to fit in 6GB
            logging_steps=10,
            max_steps=100, # Per tier
        )
        
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[soc_reward_func],
            args=training_args,
            train_dataset=dataset,
        )
        
        trainer.train()
        print(f"Tier {tier} complete. Saving checkpoint...")
        model.save_pretrained(f"{OUTPUT_DIR}_final")

if __name__ == "__main__":
    # Ensure server is running in another terminal
    train()
