import os
import json
import urllib.request
import urllib.error
import textwrap
from typing import Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")  # fallback to testing key
API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
SOC_ENV_URL = os.getenv("SOC_ENV_URL", "http://localhost:7860")
BENCHMARK = "soc-env"

DEBUG_LOG_FILE = "inference_debug.log"

def debug_log(msg: str):
    try:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI SOC analyst. Your goal: identify compromised devices, remediate, and escalate all incidents.

    MANDATORY DECISION TREE — follow every step in order:
    1. POLL (only at start or after a major action): Use the org-wide snapshot to spot suspicious devices
       (look for alert_flags: flagged_ips, unusual_outbound, high_cpu combined with outbound).
    2. INVESTIGATE: Call investigate_device on the most suspicious device you haven't investigated yet.
       Check progress.investigated_devices to avoid re-investigating.
    3. REVIEW telemetry in your head: Look at active_processes, outbound_ips, dns_queries.
       - Suspicious process names, unknown outbound IPs, or malicious dns domains = compromised.
       - No suspicious indicators = call mark_safe and move to next device.
    4. REMEDIATE if compromised (pick the right action based on evidence):
       - kill_process  → if there is a suspicious/unknown process running
       - block_ip      → if there is a suspicious IP in outbound_ips
       - isolate_device → if the device is severely compromised and you want to cut it off
    5. ESCALATE ← THIS IS MANDATORY after EVERY remediation. ALWAYS call escalate immediately after
       any kill_process / block_ip / isolate_device. Without escalate the episode never ends.
       Pick anomaly_type: data_exfiltration | malware_process | brute_force_login | suspicious_url_click | port_scan | resource_abuse
    6. LOOP: Check progress.escalated_devices. If other suspicious devices remain, go back to step 2.
       If all suspicious devices are escalated or marked safe, you are done.

    RULES:
    - NEVER poll_org more than once without a substantive action in between (wastes steps, lowers score).
    - NEVER investigate a device already in progress.investigated_devices unless you have new evidence.
    - ALWAYS escalate after remediating. The last_action_result will tell you the exact next step.
    - Read progress carefully every step — it shows exactly where you are in the workflow.

    Valid actions (respond with ONLY valid JSON, no prose, no markdown):
    {"action": "poll_org"}
    {"action": "investigate_device", "target_device": "<id>"}
    {"action": "isolate_device", "target_device": "<id>"}
    {"action": "block_ip", "target_device": "<id>", "target_ip": "<ip>"}
    {"action": "kill_process", "target_device": "<id>", "target_process": "<name>"}
    {"action": "escalate", "target_device": "<id>", "anomaly_type": "<type>", "confidence": 0.0-1.0}
    {"action": "mark_safe", "target_device": "<id>"}
    Optionally add "reasoning": "..." to explain your decision.
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
    debug_log(f"\n{'='*50}\nStarting Task: {task_name}\n{'='*50}")
    
    try:
        debug_log(f"Resetting environment for {task_name}...")
        req = urllib.request.Request(f"{SOC_ENV_URL}/reset?task={task_name}", method="POST")
        with urllib.request.urlopen(req) as resp:
            obs = json.loads(resp.read().decode("utf-8"))
        debug_log(f"Initial Observation: {json.dumps(obs, indent=2)}")
    except Exception as e:
        debug_log(f"Failed to reset environment: {e}")
        log_end(success=False, steps=0, score=0.00, rewards=[])
        return

    step = 0
    rewards = []
    done = False
    action_log = []
    
    while not done and step < max_steps:
        step += 1
        debug_log(f"\n--- Step {step} ---")
        
        prompt_content = json.dumps(obs)
        
        history_block = ""
        if action_log:
            history_block = "PAST ACTIONS YOU TOOK IN THIS EPISODE:\n" + "\n".join(action_log) + "\n\n"
            
        # To save tokens, we only send the instructions, a lean action history, and current state!
        progress = obs.get("progress") or {}
        progress_block = (
            f"EPISODE PROGRESS (use this to decide your next action):\n"
            f"  Step: {progress.get('step_number', '?')}/{progress.get('max_steps', '?')} | "
            f"Investigated: {progress.get('investigated_devices', [])} | "
            f"Escalated: {progress.get('escalated_devices', [])} | "
            f"Marked Safe: {progress.get('marked_safe_devices', [])}\n\n"
        )
        messages = [{
            "role": "user", 
            "content": f"INSTRUCTIONS:\n{SYSTEM_PROMPT}\n\n{progress_block}{history_block}CURRENT STATE:\n{prompt_content}"
        }]
        
        try:
            debug_log(f"Calling LLM ({MODEL_NAME})...")
            llm_resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=200
            )
            action_text = llm_resp.choices[0].message.content.strip()
            debug_log(f"Raw LLM Response:\n{action_text}")
            
            # Remove possible markdown formatting
            if action_text.startswith("```json"):
                action_text = action_text.replace("```json", "", 1)
                if action_text.endswith("```"):
                    action_text = action_text[:-3]
                action_text = action_text.strip()
            elif action_text.startswith("```"):
                action_text = action_text.replace("```", "", 1)
                if action_text.endswith("```"):
                    action_text = action_text[:-3]
                action_text = action_text.strip()
                
            debug_log(f"Cleaned Action Text to Parse:\n{action_text}")
            action_dict = json.loads(action_text)
            action_str = action_dict.get("action", "unknown")
            
        except Exception as e:
            debug_log(f"Exception during LLM call or JSON parsing: {str(e)}")
            action_text = '{"action": "poll_org"}'
            action_str = "invalid_format"
            action_dict = {"action": "poll_org"} # fallback
            
        try:
            req = urllib.request.Request(f"{SOC_ENV_URL}/step", method="POST")
            req.add_header("Content-Type", "application/json")
            data = json.dumps(action_dict).encode("utf-8")
            
            try:
                with urllib.request.urlopen(req, data=data) as resp:
                    step_data = json.loads(resp.read().decode("utf-8"))
                    status_code = resp.status
            except urllib.error.HTTPError as e:
                step_data = json.loads(e.read().decode("utf-8"))
                status_code = e.code

            obs = step_data.get("observation", {})
            reward = step_data.get("reward", {}).get("step_reward", 0.0)
            final_score = step_data.get("reward", {}).get("final_score")
            done = step_data.get("done", False)
            
            error_msg = None
            if status_code >= 400:
                error_msg = step_data.get("detail", "Error")
            elif "Error" in obs.get("last_action_result", ""):
                error_msg = obs["last_action_result"]
                
            result_str = obs.get("last_action_result", "")
            action_log.append(f"Step {step}: You output {action_text} -> Result was: {result_str}")
            
            rewards.append(reward)
            log_step(step, action_str, reward, done, error_msg)
            
        except Exception as e:
            rewards.append(0.0)
            log_step(step, action_str, 0.0, done=True, error=str(e))
            done = True
            final_score = 0.0

    if final_score is None:
        try:
            req = urllib.request.Request(f"{SOC_ENV_URL}/state", method="GET")
            with urllib.request.urlopen(req) as resp:
                state_data = json.loads(resp.read().decode("utf-8"))
            final_score = state_data.get("final_score", 0.0)
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
