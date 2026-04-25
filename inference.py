import os
import json
import urllib.request
import urllib.error
import textwrap
import concurrent.futures
import re
import argparse
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # Load variables from .env if it exists

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
    - NEVER call poll_org after step 1 unless you have zero information about remaining devices. If you have already seen the org snapshot and know which devices are warning/critical, go directly to investigate_device on the next uninvestigated one.
    - NEVER investigate a device already in progress.investigated_devices unless you have new evidence.
    - ALWAYS escalate after remediating. The last_action_result will tell you the exact next step.
    - Read progress carefully every step — it shows exactly where you are in the workflow.
    - After escalating a device, NEVER investigate a device that has zero alert_flags in the org snapshot. Only investigate devices with at least one alert_flag.

    Valid actions (respond with ONLY ONE valid JSON object per turn, no prose, no markdown):
    {"action": "poll_org", "reasoning": "..."}
    {"action": "investigate_device", "target_device": "<id>", "reasoning": "..."}
    {"action": "isolate_device", "target_device": "<id>", "reasoning": "..."}
    {"action": "block_ip", "target_device": "<id>", "target_ip": "<ip>", "reasoning": "..."}
    {"action": "kill_process", "target_device": "<id>", "target_process": "<name>", "reasoning": "..."}
    {"action": "escalate", "target_device": "<id>", "anomaly_type": "<type>", "confidence": 0.0-1.0, "reasoning": "..."}
    {"action": "mark_safe", "target_device": "<id>", "reasoning": "..."}
    You MUST include "reasoning" in every JSON response explaining your decision.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, combined_reward: float, done: bool, error: Optional[str], meta_reward: float = None, meta_reason: str = "") -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    meta_str = f" meta_reward={meta_reward:.2f} combined_reward={combined_reward:.2f} reason='{meta_reason}'" if meta_reward is not None else f" combined_reward={combined_reward:.2f}"
    print(f"[STEP] step={step} action={action} reward={reward:.2f}{meta_str} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Apply strict [0.01, 0.99] clamp to total reward to meet OpenEnv bounds
    total_reward = max(0.01, min(0.99, sum(rewards)))
    print(f"[END] success={success_val} steps={steps} score={score:.2f} total_reward={total_reward:.2f} rewards={rewards_str}", flush=True)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def finalize_and_log_step(step, action_str, reward, done, error_msg, client, action_text, state_obs, primary_reasoning, demo=False, cumulative_reward=0.0):
    meta_reward = 0.50
    meta_reason = "timeout or error"
    try:
        meta_reward, meta_reason = oversight_agent(client, action_text, state_obs, primary_reasoning)
    except Exception as e:
        meta_reason = f"error: {str(e)}"
    
    # Change 2: Elevate Oversight Agent weight to 0.25
    # Apply strict [0.01, 0.99] clamp to combined reward (after adding oversight bonus)
    combined_reward = max(0.01, min(0.99, reward + (meta_reward * 0.25)))
    
    # Change 2: Oversight Verdict with one sentence reason
    verdict = "APPROVED" if meta_reward >= 0.70 else "CAUTION" if meta_reward >= 0.40 else "REJECTED"
    print(f"[OVERSIGHT] action={action_str} meta_reward={meta_reward:.2f} verdict={verdict} reason='{meta_reason}'", flush=True)
    
    # Change 4: Demo Mode output
    if demo:
        target = "N/A"
        try:
            target = json.loads(action_text).get("target_device", "N/A")
        except: pass
        print(f"[STEP {step}] action={action_str} target={target} | oversight={verdict}({meta_reward:.2f}) | step_reward={reward:+.2f} | cumulative={cumulative_reward:+.2f}", flush=True)

    log_step(step, action_str, reward, combined_reward, done, error_msg, meta_reward=meta_reward, meta_reason=meta_reason)
    return combined_reward

def oversight_agent(client: OpenAI, action_text: str, state_obs: dict, agent_reasoning: str) -> tuple[float, str]:
    target_dev = "Unknown"
    act_name = "Unknown"
    try:
        ad = json.loads(action_text)
        act_name = ad.get("action", "Unknown")
        target_dev = ad.get("target_device", "Unknown")
    except:
        pass
        
    telemetry = "None"
    obs_data = state_obs.get("data", {})
    if isinstance(obs_data, dict) and "device_id" in obs_data and obs_data.get("snapshot_type") != "org_wide":
        flagged_ips = obs_data.get("network", {}).get("flagged_ips", [])
        active_processes = obs_data.get("active_processes", [])
        bytes_out = obs_data.get("network", {}).get("bytes_out_mb", 0)
        telemetry = f"Flagged IPs: {flagged_ips}, Processes: {active_processes}, Bytes Out: {bytes_out} MB"

    prompt = textwrap.dedent(f"""
    You are a senior SOC reviewer. Score the junior analyst's last 
    action on a scale of 0.0 to 1.0. Return ONLY a JSON: 
    {{"meta_reward": <float between 0.0 and 1.0>, "reason": "<one sentence>"}}.
    Penalize: repeated investigate_device, poll_org after step 1,
    actions on wrong devices. Reward: correct remediation matching
    the observed anomaly indicators, efficient escalation.
    
    Context:
    Action Taken: {act_name}
    Target Device: {target_dev}
    Device Telemetry: {telemetry}
    Analyst Reasoning: {agent_reasoning}
    
    State Snapshot:
    {json.dumps(state_obs)[:1000]}
    """).strip()
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            timeout=5.0
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
        elif content.startswith("```"):
            content = content.replace("```", "", 1)
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
            
        data = json.loads(content)
        return float(data.get("meta_reward", 0.50)), data.get("reason", "timeout or parse error")
    except Exception as e:
        return 0.50, f"error: {str(e)}"

def run_task(client: OpenAI, task_name: str, max_steps: int, demo: bool = False):
    # ── hardening: step counter MUST be initialised here, inside run_task ──
    step = 0
    rewards = []
    done = False
    action_log = []
    last_org_snapshot = None
    final_score = None
    
    # State tracking dictionary for remediation enforcement
    # {device_id: {kill_done, block_done, isolate_done, escalate_done}}
    remediation_state = {}

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    debug_log(f"\n{'='*50}\nStarting Task: {task_name}\n{'='*50}")
    
    try:
        debug_log(f"Resetting environment for {task_name}...")
        req = urllib.request.Request(f"{SOC_ENV_URL}/reset?task={task_name}", method="POST")
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            debug_log(f"Reset {task_name}: status={status} body_len={len(body)}")
            obs = json.loads(body)
        debug_log(f"Initial Observation: {json.dumps(obs, indent=2)}")
    except Exception as e:
        import traceback
        debug_log(f"Failed to reset environment for {task_name}: {str(e)}")
        debug_log(f"Reset failed for {task_name}:\n{traceback.format_exc()}")
        log_end(success=False, steps=0, score=0.00, rewards=[])
        return

    if obs.get("data", {}).get("snapshot_type") == "org_wide":
        last_org_snapshot = obs

    # ── process whitelist used by DECISION REQUIRED block ──
    PROC_WHITELIST = [
        "svchost.exe", "systemd", "init", "kernel", "explorer.exe",
        "bash", "python", "nginx", "apache2", "postgres", "chrome",
        "rsync", "tar", "cron", "python3", "node", "java"
    ]

    while not done and step < max_steps:
        try:
            step += 1
            debug_log(f"\n--- Step {step} ---")
            
            old_obs = obs
            
            # Update last_org_snapshot whenever we receive an org-wide snapshot
            if obs.get("data", {}).get("snapshot_type") == "org_wide":
                last_org_snapshot = obs

            # ── Bug 1 fix: strictly telemetry-driven DECISION REQUIRED block ──
            decision_required_block = ""
            current_anomaly_type = "resource_abuse" # fallback
            obs_data = obs.get("data", {})
            
            current_device_id = obs_data.get('device_id')
            device_state = remediation_state.get(current_device_id, {
                "kill_done": False, "block_done": False, 
                "isolate_done": False, "escalate_done": False
            })

            if isinstance(obs_data, dict) and current_device_id and obs_data.get("snapshot_type") != "org_wide":
                # --- derive indicators ---
                active_procs = obs_data.get("active_processes", [])
                sus_procs = [
                    p for p in active_procs
                    if not any(w in p.lower() for w in PROC_WHITELIST)
                ]
                has_suspicious_process = len(sus_procs) > 0

                raw_flagged = obs_data.get("network", {}).get("flagged_ips", [])
                if isinstance(raw_flagged, str):
                    import ast
                    try:
                        raw_flagged = ast.literal_eval(raw_flagged)
                    except Exception:
                        raw_flagged = [x.strip() for x in raw_flagged.split(",") if x.strip()]
                if not isinstance(raw_flagged, list):
                    raw_flagged = []
                valid_ips = [
                    ip for ip in raw_flagged
                    if ip.strip() and ip.lower() not in ["none", "n/a", "null", "[]"]
                ]
                # BUG 1 guard: must have len > 0
                has_flagged_ips = len(valid_ips) > 0

                bytes_out_mb = obs_data.get("network", {}).get("bytes_out_mb", 0)
                has_high_outbound = bytes_out_mb > 100

                # --- infer anomaly type ---
                if has_suspicious_process and not has_flagged_ips:
                    current_anomaly_type = "malware_process"
                elif has_flagged_ips and has_high_outbound:
                    current_anomaly_type = "data_exfiltration"
                elif has_suspicious_process and has_flagged_ips:
                    current_anomaly_type = "suspicious_url_click"
                else:
                    current_anomaly_type = "resource_abuse"

                decision_required_block = (
                    f"\nDECISION REQUIRED:\n"
                    f"You are currently investigating {current_device_id}.\n"
                )

                if has_suspicious_process or has_flagged_ips:
                    decision_required_block += "REQUIRED REMEDIATION SEQUENCE — complete ALL steps before escalating:\n"
                    letters = ["A", "B", "C", "D"]
                    idx = 0

                    # Step A: kill_process only if suspicious process found and not already done
                    if has_suspicious_process and not device_state["kill_done"]:
                        decision_required_block += f" Step {letters[idx]}: kill_process: {sus_procs[0]}\n"
                        idx += 1

                    # Step B: block_ip ONLY if flagged_ips AND high outbound and not already done
                    if has_flagged_ips and has_high_outbound and not device_state["block_done"]:
                        decision_required_block += f" Step {letters[idx]}: block_ip: {valid_ips[0]}\n"
                        idx += 1

                    # Step C: isolate_device if compromise confirmed and not already done
                    if not device_state["isolate_done"]:
                        decision_required_block += f" Step {letters[idx]}: isolate_device [if compromise confirmed]\n"
                        idx += 1

                    # Step D: always escalate if not already done
                    if not device_state["escalate_done"]:
                        decision_required_block += f" Step {letters[idx]}: escalate with anomaly_type={current_anomaly_type}\n"
                else:
                    decision_required_block += "- No malicious indicators found. ACTION: mark_safe\n"

                decision_required_block += "Do NOT skip any required remediation steps. ONE JSON object per turn.\n\n"

                obs["available_actions"] = [
                    "isolate_device", "block_ip", "kill_process",
                    "escalate", "mark_safe", "poll_org", "investigate_device"
                ]

            prompt_content = json.dumps(obs)
            
            history_block = ""
            if action_log:
                history_block = "PAST ACTIONS YOU TOOK IN THIS EPISODE:\n" + "\n".join(action_log) + "\n\n"

            # Read progress from the CURRENT obs (before the step fires)
            progress = obs.get("progress") or {}
            escalated = progress.get("escalated_devices", [])
            investigated = progress.get("investigated_devices", [])
                
            # ── Bug 3 fix: Filter uninvestigated list to only include flagged devices ──
            escalation_check = ""
            last_res = obs.get("last_action_result", "")
            if (
                ("escalated" in last_res.lower() or "marked safe" in last_res.lower())
                and "successfully" in last_res.lower()
            ):
                if last_org_snapshot and "devices" in last_org_snapshot.get("data", {}):
                    org_devices = last_org_snapshot["data"]["devices"]
                    flagged_uninvestigated = []
                    for dev_id, dev_info in org_devices.items():
                        flags = dev_info.get("alert_flags", [])
                        # exclude already escalated OR already investigated devices
                        # ONLY include if alert_flags is non-empty (Fix 3)
                        if flags and dev_id not in escalated and dev_id not in investigated:
                            flagged_uninvestigated.append((dev_id, flags))
                    
                    # Sort by flag count descending (Fix 3)
                    flagged_uninvestigated.sort(key=lambda x: len(x[1]), reverse=True)

                    if flagged_uninvestigated:
                        devices_str = ", ".join([f"{d} ({', '.join(f)})" for d, f in flagged_uninvestigated])
                        escalation_check = (
                            f"\nESCALATION RECORDED. Remaining uninvestigated devices with "
                            f"alert flags: {devices_str}.\n"
                            f"Investigate the highest-risk one next. Do NOT poll_org.\n\n"
                        )
                    else:
                        escalation_check = (
                            "\nESCALATION RECORDED. No remaining uninvestigated devices with alert flags.\n\n"
                        )

            # ── Fix 2 implementation via dictionary-driven prompt overrides ──
            prompt_override = ""
            if current_device_id and current_device_id in remediation_state:
                st = remediation_state[current_device_id]
                if st["isolate_done"] and not st["escalate_done"]:
                    prompt_override = (
                        f"\n⚠️ ESCALATE NOW — Device {current_device_id} has been isolated. "
                        f"You MUST call escalate on this device before doing anything else. "
                        f"This is mandatory. Do not investigate, poll, or mark_safe.\n"
                        f"Action required: escalate with target_device={current_device_id} "
                        f"anomaly_type={current_anomaly_type} confidence=0.9\n"
                    )
                elif (st["kill_done"] or st["block_done"]) and not st["isolate_done"]:
                    prompt_override = (
                        f"\nNext required action: isolate_device on {current_device_id} before escalating.\n"
                    )

            progress_block = (
                f"EPISODE PROGRESS (use this to decide your next action):\n"
                f"  Step: {progress.get('step_number', '?')}/{progress.get('max_steps', '?')} | "
                f"Investigated: {progress.get('investigated_devices', [])} | "
                f"Escalated: {escalated} | "
                f"Marked Safe: {progress.get('marked_safe_devices', [])}\n"
                f"{escalation_check}"
            )
            
            final_user_prompt = (
                f"INSTRUCTIONS:\n{SYSTEM_PROMPT}\n\n"
                f"{progress_block}{history_block}{decision_required_block}"
                f"{prompt_override}"
                f"CURRENT STATE:\n{prompt_content}"
            )

            messages = [{
                "role": "user", 
                "content": final_user_prompt
            }]
            
            action_text = ""
            action_str = "parse_fallback"
            action_dict = {"action": "poll_org", "reasoning": "parse_fallback"}
            primary_reasoning = ""

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

                # ── Bug 2 fix: 4-tier parse chain ──
                parsed = None

                # Tier 1: direct parse
                try:
                    parsed = json.loads(action_text)
                except Exception:
                    pass

                # Tier 2: strip ```json / ``` fences
                if parsed is None:
                    clean = action_text
                    if "```json" in clean:
                        clean = clean.split("```json")[-1].split("```")[0]
                    elif "```" in clean:
                        clean = clean.split("```")[1] if clean.count("```") >= 2 else clean.replace("```", "")
                    try:
                        parsed = json.loads(clean.strip())
                    except Exception:
                        pass

                # Tier 3: regex extraction
                if parsed is None:
                    match = re.search(r'\{[^{}]*\}', action_text)
                    if match:
                        try:
                            parsed = json.loads(match.group(0))
                        except Exception:
                            pass

                # Tier 4: fallback
                if parsed is None:
                    debug_log(f"PARSE FAILURE raw response: {action_text}")
                    parsed = {"action": "poll_org", "reasoning": "parse_fallback"}
                    action_str = "parse_fallback"
                else:
                    action_str = parsed.get("action", "unknown")

                action_dict = parsed
                primary_reasoning = action_dict.get("reasoning", "")

            except Exception as e:
                debug_log(f"Exception during LLM call: {str(e)}")
                action_text = '{"action": "poll_org"}'
                action_str = "parse_fallback"
                action_dict = {"action": "poll_org", "reasoning": "parse_fallback"}
                primary_reasoning = ""
                
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

                # ── hardening: final_score comes from server, not computed ──
                final_score = step_data.get("reward", {}).get("final_score")
                done = step_data.get("done", False)

                # Extra safety: presence of final_score signals episode end
                if final_score is not None:
                    done = True

                debug_log(f"Step {step} result: done={done} final_score={final_score}")
                
                error_msg = None
                if status_code >= 400:
                    error_msg = step_data.get("detail", "Error")
                elif "Error" in obs.get("last_action_result", ""):
                    error_msg = obs["last_action_result"]
                
                if "Episode already done" in str(error_msg):
                    done = True
                    
                result_str = obs.get("last_action_result", "")
                
                # ── Update remediation_state after successful action ──
                target_dev = action_dict.get("target_device")
                if target_dev and "successfully" in result_str.lower():
                    if target_dev not in remediation_state:
                        remediation_state[target_dev] = {
                            "kill_done": False, "block_done": False, 
                            "isolate_done": False, "escalate_done": False
                        }
                    
                    low_res = result_str.lower()
                    if "killed" in low_res: remediation_state[target_dev]["kill_done"] = True
                    if "blocked" in low_res: remediation_state[target_dev]["block_done"] = True
                    if "isolated" in low_res: remediation_state[target_dev]["isolate_done"] = True
                    if "escalated" in low_res: remediation_state[target_dev]["escalate_done"] = True

                action_log.append(f"Step {step}: You output {action_text} -> Result was: {result_str}")
                
                combined_r = finalize_and_log_step(
                    step, action_str, reward, done, error_msg,
                    client, action_text, old_obs, primary_reasoning,
                    demo=demo, cumulative_reward=sum(rewards)
                )
                rewards.append(combined_r)
                
            except Exception as e:
                debug_log(f"Step {step} HTTP exception: {str(e)}")
                combined_r = finalize_and_log_step(
                    step, action_str, 0.0, True, str(e),
                    client, action_text, old_obs, primary_reasoning,
                    demo=demo, cumulative_reward=sum(rewards)
                )
                rewards.append(combined_r)
                done = True
                final_score = 0.0

        except Exception as e:
            import traceback
            debug_log(f"[CRITICAL ERROR] Exception during run_task loop in {task_name}:\n{traceback.format_exc()}")
            break

    # ── hardening: final_score from server; only fall back to GET /state if None ──
    if final_score is None:
        try:
            req = urllib.request.Request(f"{SOC_ENV_URL}/state", method="GET")
            with urllib.request.urlopen(req) as resp:
                state_data = json.loads(resp.read().decode("utf-8"))
            final_score = state_data.get("final_score", 0.0)
        except Exception:
            final_score = 0.0

    success = final_score is not None and final_score >= 0.75
    log_end(success, step, final_score if final_score is not None else 0.0, rewards)

def main():
    parser = argparse.ArgumentParser(description="SOC-Env Inference Baseline")
    parser.add_argument("--task", type=str, help="Run a specific task (task_1, task_2, task_3, task_4)")
    parser.add_argument("--demo", action="store_true", help="Enable rich UX demo mode")
    args = parser.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    # Change 5: Added task_4 to the task list
    all_tasks = [
        ("task_1", 12),
        ("task_2", 15),
        ("task_3", 20),
        ("task_4", 25)
    ]
    
    if args.task:
        selected_tasks = [(t, steps) for t, steps in all_tasks if t == args.task]
    else:
        selected_tasks = all_tasks

    for task_name, max_steps in selected_tasks:
        try:
            run_task(client, task_name, max_steps, demo=args.demo)
        except Exception as e:
            import traceback
            debug_log(f"run_task crashed for {task_name}:\n{traceback.format_exc()}")
            log_end(False, 0, 0.0, [])

if __name__ == "__main__":
    main()
    executor.shutdown(wait=True)
