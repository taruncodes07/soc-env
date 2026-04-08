from typing import Tuple, Dict, Any, Optional
from server.models import (
    EpisodeState, Action, Observation, OrgSnapshot, DeviceSummary, Reward, ActionRecord,
    EpisodeProgress
)
from server.scenario_loader import load_scenario
from server.grader import TaskGrader
from datetime import datetime

class Environment:
    def __init__(self):
        self.state: Optional[EpisodeState] = None
        self.grader: Optional[TaskGrader] = None

    def _build_progress(self) -> EpisodeProgress:
        return EpisodeProgress(
            step_number=self.state.step_number,
            max_steps=self.state.max_steps,
            investigated_devices=list(self.state.investigated_devices),
            escalated_devices=list(self.state.escalated_devices),
            marked_safe_devices=list(self.state.marked_safe_devices),
        )

    def reset(self, task_id: str) -> Observation:
        self.state = load_scenario(task_id)
        self.grader = TaskGrader(self.state.ground_truth)
        
        return self._get_org_snapshot("Episode started.")

    def _get_org_snapshot(self, last_result: str) -> Observation:
        device_summaries = []
        for dev_id, dev_state in self.state.devices.items():
            dt = dev_state.telemetry
            
            # Formulate status and flags for org snapshot
            status = "healthy"
            flags = []
            
            if dt.cpu_percent > 80.0:
                flags.append("high_cpu")
            if dt.network.bytes_out_mb > 100.0:
                flags.append("unusual_outbound")
            if dt.failed_login_attempts > 0:
                flags.append("login_failures")
            if len(dt.network.flagged_ips) > 0:
                flags.append("flagged_ips")
                
            if len(flags) >= 2 or dt.cpu_percent > 90.0:
                status = "critical"
            elif len(flags) == 1 or dt.cpu_percent > 70.0:
                status = "warning"
                
            summary = DeviceSummary(
                device_id=dev_id,
                hostname=dt.hostname,
                status=status,
                cpu_percent=dt.cpu_percent,
                memory_percent=dt.memory_percent,
                bytes_out_mb=dt.network.bytes_out_mb,
                active_connections=dt.network.active_connections,
                alert_flags=flags
            )
            device_summaries.append(summary)
            
        snap = OrgSnapshot(
            timestamp=datetime.utcnow().isoformat() + "Z",
            devices=device_summaries,
            step_number=self.state.step_number
        )
        
        return Observation(
            data=snap,
            last_action_result=last_result,
            available_actions=["poll_org", "investigate_device", "isolate_device", "block_ip", "kill_process", "escalate", "mark_safe"],
            episode_done=self.state.done,
            progress=self._build_progress()
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.state.step_number += 1
        
        result_msg = ""
        obs_data = None
        step_reward_val = 0.0
        
        # Action handling
        if action.action == "poll_org":
            result_msg = "Polled org-wide snapshot."
            step_reward_val = self.grader.score_step(action, self.state)
            # Unnecessary polls are penalized slightly
            if self.state.step_number > 1:
                step_reward_val -= 0.01 
            
        elif action.action in ["investigate_device", "isolate_device", "block_ip", "kill_process", "escalate", "mark_safe"]:
            dev_id = action.target_device
            if not dev_id or dev_id not in self.state.devices:
                result_msg = f"Error: Invalid or missing target_device '{dev_id}'."
                step_reward_val = -0.02
                step_reward_val += self.grader.score_step(action, self.state)
            else:
                dev_state = self.state.devices[dev_id]
                if action.action == "investigate_device":
                    if dev_id not in self.state.investigated_devices:
                        self.state.investigated_devices.append(dev_id)
                    result_msg = f"Investigated {dev_id}. Analyze telemetry: check active_processes, outbound_ips, dns_queries. If compromised: remediate then ESCALATE. If clean: call mark_safe."
                    obs_data = dev_state.telemetry
                    step_reward_val = self.grader.score_step(action, self.state)
                
                elif action.action == "isolate_device":
                    dev_state.is_isolated = True
                    obs_data = dev_state.telemetry
                    result_msg = f"Device {dev_id} isolated from network. NEXT: Call escalate on {dev_id} with the correct anomaly_type to formally close this incident."
                    step_reward_val = self.grader.score_step(action, self.state)
                    
                elif action.action == "block_ip":
                    ip = action.target_ip
                    if ip:
                        if ip not in dev_state.blocked_ips:
                            dev_state.blocked_ips.append(ip)
                        obs_data = dev_state.telemetry
                        result_msg = f"Blocked IP {ip} on {dev_id}. NEXT: If remediation is complete, call escalate on {dev_id} with the correct anomaly_type."
                    else:
                        result_msg = "Error: Missing target_ip."
                        step_reward_val -= 0.02
                    step_reward_val += self.grader.score_step(action, self.state)
                    
                elif action.action == "kill_process":
                    proc = action.target_process
                    if proc:
                        if proc not in dev_state.killed_processes:
                            dev_state.killed_processes.append(proc)
                        obs_data = dev_state.telemetry
                        result_msg = f"Killed process '{proc}' on {dev_id}. NEXT: Call escalate on {dev_id} with the correct anomaly_type to formally close this incident."
                    else:
                        result_msg = "Error: Missing target_process."
                        step_reward_val -= 0.02
                    step_reward_val += self.grader.score_step(action, self.state)
                    
                elif action.action == "escalate":
                    if dev_id not in self.state.escalated_devices:
                        self.state.escalated_devices.append(dev_id)
                        
                        # Apply identification reward logic right away
                        gt = self.state.ground_truth
                        total_comp = len(gt.compromised_devices)
                        max_id_per_device = 0.4 / max(1, total_comp)
                        
                        if dev_id in gt.compromised_devices:
                            if action.anomaly_type == gt.anomaly_types.get(dev_id):
                                id_reward = max_id_per_device # 0.4 or 0.2
                            else:
                                id_reward = max_id_per_device / 2.0 # 0.2 or 0.1
                        else:
                            if action.anomaly_type in gt.anomaly_types.values():
                                id_reward = max_id_per_device / 4.0 # 0.1 or 0.05
                            else:
                                id_reward = 0.0
                                
                        self.state.identification_score += id_reward
                    
                    result_msg = f"Incident escalated for {dev_id} (anomaly: {action.anomaly_type}). Check progress — if other devices need investigation, continue. Otherwise mark remaining devices safe."
                    step_reward_val += self.grader.score_step(action, self.state)
                    
                elif action.action == "mark_safe":
                    if dev_id not in self.state.marked_safe_devices:
                        self.state.marked_safe_devices.append(dev_id)
                    result_msg = f"Marked device {dev_id} as safe."
                    step_reward_val += self.grader.score_step(action, self.state)
                    
        else:
            result_msg = f"Error: Unknown action '{action.action}'."
            step_reward_val = -0.02
            
        self.state.actions_taken.append(ActionRecord(step=self.state.step_number, action=action, result=result_msg))
        self.state.last_action_result = result_msg
        self.state.cumulative_reward += step_reward_val
        
        # Check done
        is_done = self.grader.is_done(self.state)
        self.state.done = is_done
        
        if is_done:
            # Episode ended, compute final rewards
            reward = self.grader.score_episode(self.state)
        else:
            reward = Reward(
                step_reward=step_reward_val,
                cumulative_reward=self.state.cumulative_reward,
                identification_score=self.state.identification_score,
                remediation_score=0.0,
                efficiency_score=0.0,
                false_positive_penalty=self.state.false_positive_penalty,
                final_score=None
            )
            
        if not obs_data:
            obs = self._get_org_snapshot(result_msg)
        else:
            obs_data.step_number = self.state.step_number
            obs = Observation(
                data=obs_data,
                last_action_result=result_msg,
                available_actions=["poll_org", "investigate_device", "isolate_device", "block_ip", "kill_process", "escalate", "mark_safe"],
                episode_done=self.state.done,
                progress=self._build_progress()
            )
            
        return obs, reward, is_done, {}
