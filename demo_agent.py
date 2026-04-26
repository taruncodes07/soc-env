"""
🛡️ SOC-Env: Demo Inference Script
————————————————————————————————
Loads the trained RL model and runs a single triage episode.
Satisfies the requirement for a clean demo/inference interface.
"""

import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from soc_openenv import make_env

# --- Config ---
MODEL_PATH = "./soc_rl_stable_final"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Fallback if not in .env
TASK_ID    = "task_1"                      # Which task to demo
MAX_STEPS  = 15

def run_demo():
    print(f"🔍 Loading Trained Agent: {MODEL_PATH}")
    
    # 1. Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    
    # 2. Attach Trained LoRA weights
    if os.path.exists(MODEL_PATH):
        print("✅ Found trained weights, attaching...")
        model = PeftModel.from_pretrained(model, MODEL_PATH)
    else:
        print("⚠️ No trained weights found. Running baseline MODEL ONLY.")

    # 3. Setup OpenEnv
    env = make_env(task_id=TASK_ID)
    obs = env.reset()
    
    print(f"\n🚀 Starting Demo Episode ({TASK_ID})")
    print("-" * 40)

    total_reward = 0.0
    for step in range(1, MAX_STEPS + 1):
        # Format the observation for the agent
        obs_data = json.loads(obs.model_dump_json())["data"]
        state_str = json.dumps(obs_data)
        
        prompt = f"SYSTEM: SOC Analyst. Reply JSON.\\nSTATE: {state_str}\\nACTION:"
        
        # Generate Action
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"\n[STEP {step}]")
        print(f"Agent Action: {response.strip()}")
        
        # Parse and Step
        try:
            import re
            match = re.search(r"(\{.*\})", response, re.DOTALL)
            action_data = json.loads(match.group(1)) if match else {}
            
            obs, reward, done, info = env.step(action_data)
            total_reward += reward
            
            print(f"Result: {obs.last_action_result}")
            print(f"Reward: {reward:.4f} | Cumulative: {total_reward:.4f}")
            
            if done:
                print("\n✅ Episode Complete!")
                break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

    print("-" * 40)
    print(f"Final Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    run_demo()
