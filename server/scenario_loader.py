import json
import os
import random
import copy
from typing import Dict, Any, List, Tuple
from server.models import (
    DeviceTelemetry, NetworkDetail, OrgSnapshot, DeviceSummary,
    EpisodeState, GroundTruth, DeviceState
)
from datetime import datetime

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")

# --- ANOMALY PROFILES ---
ANOMALY_PROFILES = {
    "data_exfiltration": {
        "cpu_boost": (5, 15),
        "bytes_out_boost": (300, 1500),
        "flagged_ips_count": 1,
        "malicious_process_type": "network_tool",
        "suspicious_dns": ["transfer.sh", "pastebin.com", "mega.nz"]
    },
    "malware_process": {
        "cpu_boost": (40, 80),
        "bytes_out_boost": (5, 50),
        "flagged_ips_count": 0,
        "malicious_process_type": "miner",
        "suspicious_dns": ["c2-server.evil.com", "pool.monero.org"]
    },
    "brute_force_login": {
        "cpu_boost": (10, 20),
        "login_fail_boost": (5, 20),
        "malicious_process_type": "auth_service",
        "suspicious_dns": ["internal.auth.attack"]
    },
    "suspicious_url_click": {
        "cpu_boost": (5, 10),
        "bytes_out_boost": (10, 100),
        "flagged_ips_count": 1,
        "malicious_process_type": "browser_extension"
    },
    "lateral_movement": {
        "cpu_boost": (10, 25),
        "internal_scan": True,
        "malicious_process_type": "shell",
        "login_fail_boost": (2, 8)
    }
}

# --- DEVICE POOLS ---
PROCESS_WHITE_LIST = [
    "svchost.exe", "systemd", "init", "kernel", "explorer.exe",
    "bash", "python", "nginx", "apache2", "postgres", "chrome",
    "rsync", "tar", "cron", "python3", "node", "java", "sshd", "docker"
]

MALICIOUS_NAMES = {
    "miner": ["kdevtmpfsi", "xmrig", "sys_update", "gpu_driver_x"],
    "network_tool": ["nmap", "nc", "ssh_tunnel", "tcp_dump"],
    "shell": ["bash_rev", "powershell_stager", "sh_temp"],
    "browser_extension": ["chrome_plugin_v2", "adblock_plus_patch"],
    "auth_service": ["login_worker", "pam_bridge", "auth_sync"]
}

def get_random_ip(internal: bool = False) -> str:
    if internal:
        return f"10.0.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"{random.randint(11,249)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"

def generate_telemetry(item: Dict, status: str, anomaly_type: str, tier: int) -> Dict:
    """
    Stochastic Telemetry Generator
    tier 1: low noise, high signal
    tier 4: high noise, subtle signal
    """
    role = item.get("role", "workstation")
    
    # 1. Base Process Generation
    num_procs = random.randint(5, 12) if tier < 3 else random.randint(10, 25)
    procs = random.sample(PROCESS_WHITE_LIST, k=min(num_procs, len(PROCESS_WHITE_LIST)))
    
    # 2. Resource Logic
    base_cpu = 5.0 if role == "workstation" else 15.0 if role == "server" else 10.0
    base_mem = 20.0 if role == "workstation" else 45.0 if role == "server" else 25.0
    
    cpu = base_cpu + (len(procs) * 0.5) + random.uniform(0, 5)
    mem = base_mem + (len(procs) * 1.2) + random.uniform(0, 10)
    disk = random.uniform(10, 80)
    
    login_fails = 0
    bytes_in = random.uniform(1.0, 50.0)
    bytes_out = random.uniform(0.5, 10.0)
    flagged_ips = []
    dns = ["google.com", "microsoft.com", "local.internal"]
    
    # 3. Anomaly Injection
    malicious_proc = None
    if status != "healthy" and anomaly_type in ANOMALY_PROFILES:
        prof = ANOMALY_PROFILES[anomaly_type]
        
        # Subtle signal in higher tiers
        signal_mult = 1.0 if tier < 3 else 0.5
        
        cpu += random.uniform(*prof.get("cpu_boost", (0,0))) * signal_mult
        bytes_out += random.uniform(*prof.get("bytes_out_boost", (0,0))) * signal_mult
        login_fails += int(random.uniform(*prof.get("login_fail_boost", (0,0))) * signal_mult)
        
        if prof.get("flagged_ips_count", 0) > 0:
            flagged_ips.append(get_random_ip(internal=False))
            
        if prof.get("internal_scan"):
            flagged_ips.append(get_random_ip(internal=True))
            
        if "suspicious_dns" in prof:
            dns.extend(random.sample(prof["suspicious_dns"], k=random.randint(1, len(prof["suspicious_dns"]))))
            
        # Malware process name obfuscation
        m_type = prof.get("malicious_process_type", "miner")
        raw_name = random.choice(MALICIOUS_NAMES[m_type])
        if tier >= 3:
            # Add obfuscation: typosquatting or hiding
            obf = random.choice([".exe", ".sh", "_worker", ""])
            malicious_proc = raw_name + obf
        else:
            malicious_proc = raw_name
            
        procs.append(malicious_proc)

    # 4. Noise/Decoys for Tier 3+
    if tier >= 3 and status == "healthy":
        if random.random() < 0.2: # 20% chance of decoy spike
            cpu += 30.0 # High CPU on healthy device
        if random.random() < 0.1:
            bytes_out += 200.0 # High traffic on healthy device

    return {
        "cpu_percent": min(100.0, cpu),
        "memory_percent": min(100.0, mem),
        "disk_percent": min(100.0, disk),
        "active_processes": procs,
        "network": {
            "bytes_in_mb": bytes_in,
            "bytes_out_mb": bytes_out,
            "active_connections": random.randint(2, 50),
            "destination_ports": item.get("baseline", {}).get("network", {}).get("destination_ports", [443, 80]),
            "flagged_ips": flagged_ips
        },
        "failed_login_attempts": login_fails,
        "dns_queries": list(set(dns)),
        "malicious_proc_name": malicious_proc
    }

def load_scenario(task_id: str, randomize: bool = True) -> EpisodeState:
    file_path = os.path.join(SCENARIO_DIR, f"{task_id}.json")
    if not os.path.exists(file_path):
        raise ValueError(f"Scenario {task_id} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine Tier from Task ID
    tier = 1
    if "task_2" in task_id: tier = 2
    if "task_3" in task_id: tier = 3
    if "task_4" in task_id: tier = 4

    max_steps = data.get("max_steps", 15)
    all_device_items = data["devices"]
    all_device_ids = [d["device_id"] for d in all_device_items]
    
    # --- DYNAMIC CRITICALITY ASSIGNMENT ---
    # L1: Mostly Low/Med. L4: High/Critical decoys.
    crit_options = ["low", "medium", "high", "critical"]
    if tier == 1:
        weights = [0.5, 0.4, 0.1, 0.0]
    elif tier == 2:
        weights = [0.3, 0.4, 0.2, 0.1]
    elif tier == 3:
        weights = [0.1, 0.3, 0.4, 0.2]
    else: # Tier 4
        weights = [0.0, 0.2, 0.4, 0.4]

    device_criticality = {
        did: random.choices(crit_options, weights=weights)[0] 
        for did in all_device_ids
    }

    # 1. Randomly Pick Compromised Devices
    num_to_compromise = 1
    if tier == 3: num_to_compromise = 2
    if tier == 4: num_to_compromise = 2
    
    comp_ids = random.sample(all_device_ids, k=num_to_compromise)
    
    # 2. Assign Anomaly Types
    possible_anomalies = list(ANOMALY_PROFILES.keys())
    if tier < 2: possible_anomalies = ["data_exfiltration", "malware_process"]
    
    anomaly_map = {cid: random.choice(possible_anomalies) for cid in comp_ids}
    
    # 3. Generate Telemetry for ALL devices
    devices_state: Dict[str, DeviceState] = {}
    correct_remed_dict = {}
    
    for item in all_device_items:
        dev_id = item["device_id"]
        is_comp = dev_id in comp_ids
        a_type = anomaly_map.get(dev_id, None)
        crit = device_criticality[dev_id]
        
        tele_data = generate_telemetry(item, "compromised" if is_comp else "healthy", a_type, tier)
        
        net_info = tele_data["network"]
        net_detail = NetworkDetail(
            bytes_in_mb=net_info["bytes_in_mb"],
            bytes_out_mb=net_info["bytes_out_mb"],
            active_connections=net_info["active_connections"],
            destination_ports=net_info["destination_ports"],
            flagged_ips=net_info["flagged_ips"]
        )

        telemetry = DeviceTelemetry(
            device_id=dev_id,
            hostname=item["hostname"],
            os=item["os"],
            role=item["role"],
            cpu_percent=tele_data["cpu_percent"],
            memory_percent=tele_data["memory_percent"],
            disk_percent=tele_data["disk_percent"],
            active_processes=tele_data["active_processes"],
            network=net_detail,
            logged_in_user=item["logged_in_user"],
            failed_login_attempts=tele_data["failed_login_attempts"],
            dns_queries=tele_data["dns_queries"],
            outbound_ips=net_info["flagged_ips"],
            criticality=crit,
            last_seen=datetime.utcnow().isoformat() + "Z",
            step_number=0
        )
        
        devices_state[dev_id] = DeviceState(
            telemetry=telemetry,
            is_compromised=is_comp,
            is_isolated=False,
            blocked_ips=[],
            killed_processes=[]
        )
        
        if is_comp:
            # Map anomaly types to req actions
            actions = ["isolate_device"]
            if a_type in ["malware_process", "brute_force_login", "lateral_movement"]:
                actions.insert(0, "kill_process")
            if a_type in ["data_exfiltration", "suspicious_url_click"]:
                actions.insert(0, "block_ip")
            correct_remed_dict[dev_id] = actions

    healthy_ids = [d for d in all_device_ids if d not in comp_ids]

    ground_truth = GroundTruth(
        compromised_devices=comp_ids,
        anomaly_types=anomaly_map,
        correct_remediation_actions=correct_remed_dict,
        optimal_steps=num_to_compromise * 4,
        healthy_devices=healthy_ids
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
