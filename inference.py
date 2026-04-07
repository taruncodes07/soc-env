import os
import json
import requests
import textwrap
from typing import Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
SOC_ENV_URL = os.getenv("SOC_ENV_URL", "http://localhost:7860")
BENCHMARK = "soc-env"

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI SOC analyst. You monitor an organization's device fleet for security anomalies.

    Each step you receive either:
    1. An org-wide snapshot showing all devices with high-level status
    2. A detailed telemetry report for a specific device you investigated

    You must respond with ONLY a valid JSON action object. No prose, no markdown, no explanation outside the JSON.

    Valid actions:
    - {"action": "poll_org"} — refresh the org-wide snapshot
    - {"action": "investigate_device", "target_device": "<id>"} — get full telemetry for a device
    - {"action": "isolate_device", "target_device": "<id>"} — cut device from network (irreversible)
    - {"action": "block_ip", "target_device": "<id>", "target_ip": "<ip>"} — block an IP on a device
    - {"action": "kill_process", "target_device": "<id>", "target_process": "<name>"} — terminate process
    - {"action": "escalate", "target_device": "<id>", "anomaly_type": "<type>", "confidence": 0.0-1.0} — formal incident report
    - {"action": "mark_safe", "target_device": "<id>"} — declare device clean

    Anomaly types: data_exfiltration | malware_process | brute_force_login | suspicious_url_click | port_scan | resource_abuse

    Rules:
    - Only take destructive actions (isolate, kill, block) if you have evidence from investigating the device
    - Always escalate after remediating a compromised device
    - Be efficient — unnecessary steps reduce your score
    - Include optional "reasoning" field to explain your decision
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def run_task(client: OpenAI, task_name: str, max_steps: int):
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        resp = requests.post(f"{SOC_ENV_URL}/reset?task={task_name}")
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        log_end(success=False, steps=0, score=0.00, rewards=[])
        return

    step = 0
    rewards = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    
    while not done and step < max_steps:
        step += 1
        messages.append({"role": "user", "content": json.dumps(obs)})
        
        try:
            llm_resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=200
            )
            action_text = llm_resp.choices[0].message.content.strip()
            
            # Remove possible markdown formatting if LLM ignores instruction
            if action_text.startswith("```json"):
                action_text = action_text.replace("```json", "").replace("```", "").strip()
            elif action_text.startswith("```"):
                action_text = action_text.replace("```", "").strip()
                
            action_dict = json.loads(action_text)
            action_str = action_dict.get("action", "unknown")
            messages.append({"role": "assistant", "content": action_text})
        except Exception as e:
            action_str = "invalid_format"
            action_dict = {"action": "poll_org"} # fallback to something valid
            messages.append({"role": "assistant", "content": f"Failed to generate valid JSON: {str(e)}"})
            
        try:
            step_resp = requests.post(f"{SOC_ENV_URL}/step", json=action_dict)
            step_resp.raise_for_status()
            step_data = step_resp.json()
            obs = step_data["observation"]
            reward = step_data["reward"]["step_reward"]
            final_score = step_data["reward"].get("final_score")
            done = step_data["done"]
            
            error_msg = None
            if step_resp.status_code >= 400:
                error_msg = step_data.get("detail", "Error")
            elif "Error" in obs.get("last_action_result", ""):
                error_msg = obs["last_action_result"]
                
            rewards.append(reward)
            log_step(step, action_str, reward, done, error_msg)
            
        except requests.exceptions.RequestException as e:
            rewards.append(0.0)
            log_step(step, action_str, 0.0, done=True, error=str(e))
            done = True
            final_score = 0.0

    if final_score is None:
        try:
            state_resp = requests.get(f"{SOC_ENV_URL}/state")
            final_score = state_resp.json().get("final_score", 0.0)
        except:
            final_score = 0.0
            
    success = final_score is not None and final_score >= 0.75
    log_end(success, step, final_score if final_score is not None else 0.0, rewards)

def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    tasks = [
        ("task_1", 12),
        ("task_2", 15),
        ("task_3", 20)
    ]
    
    for task_name, max_steps in tasks:
        run_task(client, task_name, max_steps)

if __name__ == "__main__":
    main()
