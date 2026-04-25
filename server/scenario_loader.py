import json
import os
import random
import copy
from typing import Dict, Any, List
from server.models import (
    DeviceTelemetry, NetworkDetail, OrgSnapshot, DeviceSummary,
    EpisodeState, GroundTruth, DeviceState
)
from datetime import datetime

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")

def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    base_copy = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_copy[key] = deep_merge(base_copy[key], value)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy

def load_scenario(task_id: str, randomize: bool = True) -> EpisodeState:
    file_path = os.path.join(SCENARIO_DIR, f"{task_id}.json")
    if not os.path.exists(file_path):
        raise ValueError(f"Scenario {task_id} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_steps = data.get("max_steps", 15)
    gt_data = data["ground_truth"]
    
    comp_devices_str = gt_data["compromised_device"]
    orig_comp_devices = comp_devices_str.split(",") if isinstance(comp_devices_str, str) else comp_devices_str
    if isinstance(orig_comp_devices, str):
        orig_comp_devices = [orig_comp_devices]
        
    anomaly_types_str = gt_data["anomaly_type"]
    anomaly_types_list = anomaly_types_str.split(",") if isinstance(anomaly_types_str, str) else [anomaly_types_str]
    
    remed_actions_list = gt_data.get("correct_remediation_actions", [])

    overlays_map = {}
    for item in data["devices"]:
        if "anomaly_overlay" in item and item["anomaly_overlay"]:
            overlays_map[item["device_id"]] = item["anomaly_overlay"]

    all_device_ids = [d["device_id"] for d in data["devices"]]
    
    comp_devices_to_use = orig_comp_devices
    process_name_map = {}
    ip_map = {}
    
    if randomize:
        num_comp = len(orig_comp_devices)
        new_comp_devices = random.sample(all_device_ids, num_comp)
        
        old_to_new_map = dict(zip(orig_comp_devices, new_comp_devices))
        new_overlays_map = {}
        for old_dev, overlay in overlays_map.items():
            if old_dev in old_to_new_map:
                new_overlays_map[old_to_new_map[old_dev]] = overlay
                
        overlays_map = new_overlays_map
        comp_devices_to_use = [old_to_new_map.get(d, d) for d in orig_comp_devices]

    anomaly_types_dict = {}
    for i, dev in enumerate(comp_devices_to_use):
        if i < len(anomaly_types_list):
            anomaly_types_dict[dev] = anomaly_types_list[i]
            
    remed_actions_dict = {}
    for i, dev in enumerate(comp_devices_to_use):
        if i < len(remed_actions_list):
            acts = remed_actions_list[i]
            if isinstance(acts, str):
                acts = acts.split(",")
            remed_actions_dict[dev] = acts

    healthy_devices = [d for d in all_device_ids if d not in comp_devices_to_use]

    ground_truth = GroundTruth(
        compromised_devices=comp_devices_to_use,
        anomaly_types=anomaly_types_dict,
        correct_remediation_actions=remed_actions_dict,
        optimal_steps=gt_data["optimal_steps"],
        healthy_devices=healthy_devices,
    )

    devices_state: Dict[str, DeviceState] = {}
    for item in data["devices"]:
        dev_id = item["device_id"]
        baseline = item["baseline"]
        overlay = overlays_map.get(dev_id)

        merged = copy.deepcopy(baseline)
        if overlay:
            merged = deep_merge(merged, overlay)

        if randomize:
            # CPU Noise
            cpu = merged.get("cpu_percent", 0.0)
            noise = random.uniform(-10.0, 10.0)
            merged["cpu_percent"] = max(0.0, min(100.0, cpu + noise))
            
            # Map IPs
            net = merged.get("network", {})
            new_flagged = []
            for ip in net.get("flagged_ips", []):
                if ip not in ip_map:
                    ip_map[ip] = f"{random.randint(10,250)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
                new_flagged.append(ip_map[ip])
            if "flagged_ips" in net:
                net["flagged_ips"] = new_flagged
                merged["network"] = net
                
            # Map Processes
            procs = merged.get("active_processes", [])
            new_procs = []
            for p in procs:
                if p not in process_name_map:
                    if p in ["svchost.exe", "explorer.exe", "systemd", "sshd"]: 
                        process_name_map[p] = p # don't scramble standard ones
                    else:
                        ext = p.split(".")[-1] if "." in p else "svc"
                        base_hash = hex(abs(hash(p)))[2:6]
                        process_name_map[p] = f"proc_{base_hash}.{ext}"
                new_procs.append(process_name_map[p])
            if "active_processes" in merged:
                merged["active_processes"] = new_procs

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
            outbound_ips=network_data.get("flagged_ips", []),
            last_seen=datetime.utcnow().isoformat() + "Z",
            step_number=0
        )

        is_compromised = dev_id in comp_devices_to_use

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
