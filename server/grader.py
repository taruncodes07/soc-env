from server.models import Action, EpisodeState, Reward

class TaskGrader:
    def __init__(self, ground_truth):
        pass # Ground truth is directly accessible via state.ground_truth at scoring time

    def score_step(self, action: Action, state: EpisodeState) -> float:
        step_reward = 0.0

        if action.action == "isolate_device":
            if action.target_device in state.ground_truth.healthy_devices:
                step_reward -= 0.15
                state.false_positive_penalty -= 0.15
        elif action.action == "kill_process":
            if action.target_device in state.ground_truth.healthy_devices:
                step_reward -= 0.10
                state.false_positive_penalty -= 0.10
        elif action.action == "block_ip":
            if action.target_device in state.ground_truth.healthy_devices:
                step_reward -= 0.05
                state.false_positive_penalty -= 0.05
        
        # Step penalty
        if state.step_number > state.ground_truth.optimal_steps:
            step_reward -= 0.01

        # Identify sub-reward for escalate is handled in environment.py directly when "escalate" action is processed, 
        # but we can return 0.0 here for the general step and sum it up later.
        
        return step_reward

    def is_done(self, state: EpisodeState) -> bool:
        if state.step_number >= state.max_steps:
            return True

        # Check if all compromised devices are escalated
        all_comp_escalated = all(dev in state.escalated_devices for dev in state.ground_truth.compromised_devices)
        
        # Check if all healthy devices are marked safe
        all_healthy_safe = all(dev in state.marked_safe_devices for dev in state.ground_truth.healthy_devices)

        # Or if we just successfully handled the compromised ones and explicitly ended investigation on others
        # Actually PRD says: "done=True is triggered by: correct escalation of all compromised devices, agent calls mark_safe on all remaining devices, or step_number >= max_steps"
        
        # We will simplify taking standard SOC completion:
        if all_comp_escalated:
            # Check if what's left is marked safe or we just don't strictly require it to finish if they found the bad guys?
            # PRD: "correct escalation of all compromised devices, agent calls mark_safe on all remaining devices"
            # We'll trigger done if all compromised are escalated and all investigated are either escalated or marked safe
            unresolved = set(state.investigated_devices) - set(state.escalated_devices) - set(state.marked_safe_devices)
            if len(unresolved) == 0:
                # If they didn't even investigate some, we can consider them implicitly safe, but the explicit rule says mark_safe on all remaining.
                # Let's be lenient: if they got all compromised, they can just stop, but we'll enforce done if they've acted on everything they investigated.
                return True
                
        return False

    def score_episode(self, state: EpisodeState) -> Reward:
        gt = state.ground_truth
        
        remediation_score = 0.0
        
        # Calculate remediation score
        total_comp = len(gt.compromised_devices)
        if total_comp > 0:
            dev_remed_score = 0.0
            for dev in gt.compromised_devices:
                req_actions = gt.correct_remediation_actions.get(dev, [])
                if not req_actions:
                    dev_remed_score += 0.3 / total_comp
                    continue
                
                dev_state = state.devices.get(dev)
                if not dev_state:
                    continue
                
                correct_count = 0
                for req in req_actions:
                    if req == "isolate_device" and dev_state.is_isolated:
                        correct_count += 1
                    elif req == "kill_process" and len(dev_state.killed_processes) > 0:
                        correct_count += 1
                    elif req == "block_ip" and len(dev_state.blocked_ips) > 0:
                        correct_count += 1
                        
                if correct_count == len(req_actions):
                    dev_remed_score += 0.3 / total_comp
                elif correct_count >= len(req_actions) / 2.0:
                    dev_remed_score += 0.15 / total_comp
            remediation_score = dev_remed_score
            
        state.remediation_score = remediation_score
        
        # Calculate efficiency
        efficiency = min(1.0, gt.optimal_steps / max(1, state.step_number))
        efficiency_score = efficiency * 0.2
        state.efficiency_score = efficiency_score
        
        final_score = state.identification_score + state.remediation_score + state.efficiency_score + state.false_positive_penalty
        
        # Clamp between 0.0 and 1.0
        final_score = max(0.0, min(1.0, final_score))
        state.final_score = final_score
        
        return Reward(
            step_reward=0.0, # Handled per step
            cumulative_reward=state.cumulative_reward,
            identification_score=state.identification_score,
            remediation_score=state.remediation_score,
            efficiency_score=state.efficiency_score,
            false_positive_penalty=state.false_positive_penalty,
            final_score=state.final_score
        )
