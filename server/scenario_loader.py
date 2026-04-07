import json
import os
from typing import Dict, Any, List
from server.models import (
    DeviceTelemetry, NetworkDetail, OrgSnapshot, DeviceSummary,
    EpisodeState, GroundTruth, DeviceState
)
from datetime import datetime

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")

def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    base_copy = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_copy[key] = deep_merge(base_copy[key], value)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy

def load_scenario(task_id: str) -> EpisodeState:
    file_path = os.path.join(SCENARIO_DIR, f"{task_id}.json")
    if not os.path.exists(file_path):
        raise ValueError(f"Scenario {task_id} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_steps = data.get("max_steps", 15)

    gt_data = data["ground_truth"]
    
    comp_devices_str = gt_data["compromised_device"]
    comp_devices = comp_devices_str.split(",") if isinstance(comp_devices_str, str) else comp_devices_str
    if isinstance(comp_devices, str):
        comp_devices = [comp_devices]
        
    anomaly_types_str = gt_data["anomaly_type"]
    anomaly_types_list = anomaly_types_str.split(",") if isinstance(anomaly_types_str, str) else [anomaly_types_str]
    
    anomaly_types_dict = {}
    for i, dev in enumerate(comp_devices):
        if i < len(anomaly_types_list):
            anomaly_types_dict[dev] = anomaly_types_list[i]
            
    remed_actions_list = gt_data.get("correct_remediation_actions", [])
    remed_actions_dict = {}
    for i, dev in enumerate(comp_devices):
        if i < len(remed_actions_list):
            acts = remed_actions_list[i]
            if isinstance(acts, str):
                acts = acts.split(",")
            remed_actions_dict[dev] = acts

    healthy_devices = [d["device_id"] for d in data["devices"] if d["device_id"] not in comp_devices]

    ground_truth = GroundTruth(
        compromised_devices=comp_devices,
        anomaly_types=anomaly_types_dict,
        correct_remediation_actions=remed_actions_dict,
        optimal_steps=gt_data["optimal_steps"],
        healthy_devices=healthy_devices,
    )

    devices_state: Dict[str, DeviceState] = {}
    for item in data["devices"]:
        dev_id = item["device_id"]
        baseline = item["baseline"]
        overlay = item.get("anomaly_overlay")

        merged = baseline
        if overlay:
            merged = deep_merge(baseline, overlay)

        network_data = merged.get("network", {})
        net_detail = NetworkDetail(
            bytes_in_mb=network_data.get("bytes_in_mb", 0.0),
            bytes_out_mb=network_data.get("bytes_out_mb", 0.0),
            active_connections=network_data.get("active_connections", 0),
            destination_ports=network_data.get("destination_ports", []),
            flagged_ips=network_data.get("flagged_ips", [])
        )

        telemetry = DeviceTelemetry(
            device_id=dev_id,
            hostname=item["hostname"],
            os=item["os"],
            role=item["role"],
            cpu_percent=merged.get("cpu_percent", 0.0),
            memory_percent=merged.get("memory_percent", 0.0),
            disk_percent=merged.get("disk_percent", 0.0),
            active_processes=merged.get("active_processes", []),
            network=net_detail,
            logged_in_user=item["logged_in_user"],
            failed_login_attempts=merged.get("failed_login_attempts", 0),
            dns_queries=merged.get("dns_queries", []),
            outbound_ips=[],
            last_seen=datetime.utcnow().isoformat() + "Z",
            step_number=0
        )

        is_compromised = dev_id in comp_devices

        devices_state[dev_id] = DeviceState(
            telemetry=telemetry,
            is_compromised=is_compromised,
            is_isolated=False,
            blocked_ips=[],
            killed_processes=[]
        )

    return EpisodeState(
        task_id=task_id,
        step_number=0,
        max_steps=max_steps,
        done=False,
        devices=devices_state,
        investigated_devices=[],
        actions_taken=[],
        escalated_devices=[],
        marked_safe_devices=[],
        cumulative_reward=0.0,
        ground_truth=ground_truth
    )
