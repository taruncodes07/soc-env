import os
import json
import requests
import textwrap
import re
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
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
    You are an AI SOC analyst. You monitor an organization's device fleet for security anomalies.

    Each step you receive either:
    1. An org-wide snapshot showing all devices with high-level status
    2. A detailed telemetry report for a specific device you investigated

    You must respond with ONLY a valid JSON action object. No markdown, no backticks, no prose before or after.
    
    Valid actions exactly as follows:
    - {"action": "poll_org"}
    - {"action": "investigate_device", "target_device": "<id>"}
    - {"action": "isolate_device", "target_device": "<id>"}
    - {"action": "block_ip", "target_device": "<id>", "target_ip": "<ip>"}
    - {"action": "kill_process", "target_device": "<id>", "target_process": "<name>"}
    - {"action": "escalate", "target_device": "<id>", "anomaly_type": "<type>", "confidence": 0.0-1.0}
    - {"action": "mark_safe", "target_device": "<id>"}

    Anomaly types: data_exfiltration | malware_process | brute_force_login | suspicious_url_click | port_scan | resource_abuse

    Rules & Step-wise Policy:
    1. If you have not yet identified the most suspicious device in the org snapshot, choose the device with the strongest anomaly signals (critical status, combination of flagged IPs, unusual outbound, suspicious processes), and investigate_device it.
    2. Prioritize devices that match the anomaly taxonomy more strongly (e.g., flagged IPs + unknown process for malware/data_exfiltration) over those that just have high CPU or noisy decoy signals.
    3. If you have strong evidence a device is compromised (e.g., unknown/suspicious process, high bytes_out with flagged IPs, brute-force patterns), then:
       - Remediate using the anomaly taxonomy (kill malicious process, block IP, isolate device) in a minimal number of steps.
       - Avoid calling isolate_device, kill_process, or block_ip more than once on the same target unless new evidence appears.
       - Once you have isolated and blocked the suspicious IP or process for a device, focus on escalation rather than repeatedly investigating the same device.
    4. After completing remediation, immediately escalate that device with the best anomaly_type and confidence.
    5. Only then consider marking devices safe or investigating lower-severity devices.
    
    Include optional "reasoning" field to explain your decision.

    Examples
    Example 1 (Investigating a device):
    {"action": "investigate_device", "target_device": "device_001", "reasoning": "Device status is critical with flagged IPs and unusual outbound traffic."}
    
    Example 2 (Escalating after remediation):
    {"action": "escalate", "target_device": "device_001", "anomaly_type": "data_exfiltration", "confidence": 0.95, "reasoning": "IP blocked and device isolated due to 450MB outbound data. Process stopped."}
""").strip()

def extract_and_validate_action(text: str) -> tuple[bool, dict]:
    """
    Returns (is_valid, action_dict).
    - is_valid: whether the LLM output produced a schema-valid action per PRD.
    - action_dict: either the validated action, or a safe fallback (e.g., poll_org).
    """
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        debug_log(f"Invalid LLM output (no JSON found): {text}")
        return False, {"action": "poll_org"}
        
    try:
        action_dict = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        debug_log(f"Invalid LLM output (JSON decode error): {e}")
        return False, {"action": "poll_org"}
        
    action = action_dict.get("action")
    valid_actions = {
        "poll_org", "investigate_device", "isolate_device", 
        "block_ip", "kill_process", "escalate", "mark_safe"
    }
    if action not in valid_actions:
        debug_log(f"Invalid LLM output (unknown action): {action}")
        return False, {"action": "poll_org"}
        
    if action in ["investigate_device", "isolate_device", "kill_process", "block_ip", "escalate", "mark_safe"]:
        if not isinstance(action_dict.get("target_device"), str):
            debug_log(f"Invalid LLM output (missing target_device for action {action})")
            return False, {"action": "poll_org"}
            
    if action == "block_ip" and not isinstance(action_dict.get("target_ip"), str):
        debug_log("Invalid LLM output (missing target_ip for block_ip)")
        return False, {"action": "poll_org"}
        
    if action == "kill_process" and not isinstance(action_dict.get("target_process"), str):
        debug_log("Invalid LLM output (missing target_process for kill_process)")
        return False, {"action": "poll_org"}
        
    if action == "escalate":
        valid_anomalies = {
            "data_exfiltration", "malware_process", "brute_force_login", 
            "suspicious_url_click", "port_scan", "resource_abuse"
        }
        if action_dict.get("anomaly_type") not in valid_anomalies:
            debug_log(f"Invalid LLM output (invalid anomaly_type for escalate)")
            return False, {"action": "poll_org"}
        try:
            float(action_dict.get("confidence", -1))
        except ValueError:
            debug_log("Invalid LLM output (invalid confidence type for escalate)")
            return False, {"action": "poll_org"}

    return True, action_dict

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
        resp = requests.post(f"{SOC_ENV_URL}/reset?task={task_name}")
        resp.raise_for_status()
        obs = resp.json()
        debug_log(f"Initial Observation: {json.dumps(obs, indent=2)}")
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        log_end(success=False, steps=0, score=0.00, rewards=[])
        return

    step = 0
    rewards = []
    done = False
    action_log = []
    applied_destructive_actions = {}
    device_state = {}
    
    while not done and step < max_steps:
        step += 1
        debug_log(f"\n--- Step {step} ---")
        
        prompt_content = json.dumps(obs)
        
        history_block = ""
        if action_log:
            history_block = "PAST ACTIONS YOU TOOK IN THIS EPISODE:\n" + "\n".join(action_log) + "\n\n"
            
        # To save tokens, we only send the instructions, a lean action history, and current state!
        messages = [{
            "role": "user", 
            "content": f"INSTRUCTIONS:\n{SYSTEM_PROMPT}\n\n{history_block}CURRENT STATE:\n{prompt_content}"
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
            
            is_valid, action_dict = extract_and_validate_action(action_text)
            
            if not is_valid:
                debug_log("Action validation failed. Using fallback action.")
                # user requested to log clearly that it was invalid format in debug log
                # action_dict is already set to {"action": "poll_org"} by the helper
            else:
                act = action_dict.get("action")
                target = action_dict.get("target_device")
                
                if act == "investigate_device":
                    if target not in device_state:
                        device_state[target] = {"investigated": True, "has_blocked_ip": False, "has_killed_suspicious": False, "isolated": False, "escalated": False}
                    else:
                        device_state[target]["investigated"] = True
                        if (device_state[target]["isolated"] or device_state[target]["has_blocked_ip"] or device_state[target]["has_killed_suspicious"]) and not device_state[target]["escalated"]:
                            debug_log(f"WARNING: Device {target} appears remediated but LLM is investigating again. Consider escalating instead.")
                elif act == "escalate":
                    if target not in device_state:
                         device_state[target] = {"investigated": False, "has_blocked_ip": False, "has_killed_suspicious": False, "isolated": False, "escalated": True}
                    else:
                         device_state[target]["escalated"] = True
                elif act in ["isolate_device", "kill_process", "block_ip"]:
                    param = action_dict.get("target_process") if act == "kill_process" else action_dict.get("target_ip") if act == "block_ip" else None
                    key = (target, act, param)
                    if key in applied_destructive_actions:
                         debug_log(f"WARNING: Repeated action '{act}' on {target} detected. Replacing with poll_org.")
                         action_dict = {"action": "poll_org"}
                         is_valid = False
                    else:
                         applied_destructive_actions[key] = True
                         if target not in device_state:
                             device_state[target] = {"investigated": False, "has_blocked_ip": False, "has_killed_suspicious": False, "isolated": False, "escalated": False}
                         if act == "isolate_device": device_state[target]["isolated"] = True
                         if act == "kill_process": device_state[target]["has_killed_suspicious"] = True
                         if act == "block_ip": device_state[target]["has_blocked_ip"] = True
            
        except Exception as e:
            err_str = str(e)
            if "402" in err_str:
                debug_log(f"Provider returned 402 Error (Credit Depleted): {err_str[:200]}. Skipping LLM and using poll_org fallback.")
            else:
                debug_log(f"Exception during LLM call or JSON parsing: {err_str}")
            action_text = '{"action": "poll_org"}'
            action_dict = {"action": "poll_org"} # fallback
            is_valid = False
            
        try:
            step_resp = requests.post(f"{SOC_ENV_URL}/step", json=action_dict)
            
            if step_resp.status_code >= 400:
                debug_log(f"Server returned error code {step_resp.status_code}: {step_resp.text}")
                log_step(step, action_dict["action"], 0.0, done=True, error=f"Server Error {step_resp.status_code}")
                # End Task Due to 4xx as per instructions
                final_score = 0.0
                break
                
            step_data = step_resp.json()
            obs = step_data["observation"]
            reward = step_data["reward"]["step_reward"]
            final_score = step_data["reward"].get("final_score")
            done = step_data["done"]
            
            error_msg = None
            if "Error" in obs.get("last_action_result", ""):
                error_msg = obs["last_action_result"]
                
            result_str = obs.get("last_action_result", "")
            action_log.append(f"Step {step}: You output {json.dumps(action_dict)} -> Result was: {result_str}")
            
            rewards.append(reward)
            log_step(step, action_dict["action"], reward, done, error_msg)
            
        except requests.exceptions.RequestException as e:
            rewards.append(0.0)
            log_step(step, action_dict["action"], 0.0, done=True, error=str(e))
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
