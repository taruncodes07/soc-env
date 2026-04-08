---
title: SOC-Env
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
---

# SOC-Env: Security Operations Center Environment

SOC-Env is an OpenEnv-compliant reinforcement learning environment that simulates a Security Operations Center (SOC) monitoring an organization's device fleet. An AI agent acts as an autonomous SOC analyst — it polls telemetry, triages anomalies, investigates suspicious devices, and executes remediation actions.

## 1. Environment Description

AI-powered security operations is one of the most commercially relevant and technically unsolved problems in enterprise software. Real SOC analysts deal with alert fatigue, multi-device correlation, and time-pressured triage decisions daily. SOC-Env provides a deterministic, reproducible, reward-shaped environment where agents can be evaluated and trained on genuine SOC reasoning tasks.

The environment simulates the following human workflow:
- A SOC analyst monitors a dashboard of all organization devices
- They identify which devices show anomalous signals
- They drill into specific devices for full telemetry
- They correlate signals across devices if needed
- They execute the correct remediation action
- They formally escalate the incident with the correct classification

## 2. Observation Space

The observation space returns structured JSON conforming to Pydantic models. Initial state and `poll_org` return an `OrgSnapshot`. Detailed investigation returns `DeviceTelemetry`.

**OrgSnapshot:**
```json
{
  "snapshot_type": "org_wide",
  "timestamp": "2024-03-24T12:00:00Z",
  "devices": [
    {
      "device_id": "device_001",
      "hostname": "WORKSTATION-ALICE",
      "status": "critical",
      "cpu_percent": 78.0,
      "memory_percent": 42.0,
      "bytes_out_mb": 450.0,
      "active_connections": 8,
      "alert_flags": ["high_cpu", "unusual_outbound"]
    }
  ],
  "step_number": 0
}
```

**DeviceTelemetry:** Includes detailed breakdown of CPU, memory, active_processes, network (bytes_in/out, ports, flagged IPs), failed logins, and dns_queries.

## 3. Action Space

Agents submit JSON actions. Valid actions:

| Action | Required Fields | Effect |
|---|---|---|
| `poll_org` | none | Returns fresh org-wide snapshot |
| `investigate_device` | `target_device` | Returns full DeviceTelemetry for one device |
| `isolate_device` | `target_device` | Cuts device from network, irreversible in episode |
| `block_ip` | `target_device`, `target_ip` | Blocks specific IP on device firewall |
| `kill_process` | `target_device`, `target_process` | Terminates named process on device |
| `escalate` | `target_device`, `anomaly_type`, `confidence` | Formally reports incident, closes device investigation |
| `mark_safe` | `target_device` | Declares device clean, closes investigation |

## 4. Tasks

1. **task_1 (Easy): Single Device Anomaly:** One device exfiltrating data clearly visible on the org snapshot.
2. **task_2 (Medium): Multi-Device Triage:** 6 devices. 3 have noisy elevated metrics (false positives), 1 has a real malware process. Agent must triage efficiently.
3. **task_3 (Hard): Correlated Multi-Stage Attack:** 8 devices. A workstation clicked a suspicious URL opening a connection to an internal server which is now exfiltrating data. Agent must correlate across multiple devices.

## 5. Reward Function [0.0 - 1.0]

- **Identification Score (max 0.4):** Awarded on escalate. 0.4 for correct device + correct anomaly. Partial for correct device + wrong anomaly, etc.
- **Remediation Score (max 0.3):** Evaluated at episode end. 0.3 for mapping required remediations (kill/block/isolate) correctly to the compromised device.
- **Efficiency Score (max 0.2):** `min(1.0, optimal_steps / actual_steps) * 0.2`
- **False Positive Penalty (min -0.3):** Applied immediately for isolating/killing/blocking on a healthy device.
- **Step Penalty:** Applied for every step beyond optimal or unnecessary org polls.

## 6. Setup Instructions

1. `pip install -r requirements.txt`
2. `python -m uvicorn server.app:app --host 0.0.0.0 --port 7860`
3. Run the baseline agent: `python inference.py` (ensure `API_BASE_URL` and `MODEL_NAME` and `HF_TOKEN` are set).

Alternatively via Docker:
`docker build -t soc-env .`
`docker run -p 7860:7860 soc-env`

## 7. Baseline Scores

Target expectations for a frontier model like Qwen2.5-72B-Instruct:
- **task_1:** 0.75 - 0.90
- **task_2:** 0.50 - 0.70
- **task_3:** 0.25 - 0.50

## 8. Anomaly Type Reference

- `data_exfiltration`: High bytes_out, flagged IPs, unknown process
- `malware_process`: Unknown process, high CPU, outbound C2 traffic
- `brute_force_login`: High failed_login_attempts, single source IP
- `suspicious_url_click`: Bad domain in dns_queries, new spawned process
- `port_scan`: High connection count, many dest ports, low data
- `resource_abuse`: CPU/memory pegged, legitimate runaway process
