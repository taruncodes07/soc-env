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

**SOC-Env** is an OpenEnv-compliant reinforcement learning environment that simulates a Security Operations Center (SOC) monitoring an organization's device fleet. An AI agent acts as an autonomous SOC analyst — it polls telemetry, triages anomalies, investigates suspicious devices, correlates cross-device signals, and executes remediation actions.

This environment presents a deterministic, reproducible, reward-shaped scenario where agents can be evaluated and trained on genuine SOC reasoning tasks such as mitigating alert fatigue and time-pressured triage decisions.

---

## 1. System Architecture & Request Flow

The system consists of a RESTful FastAPI-based OpenEnv environment and an independent AI agent logic defined in `inference.py`.

```text
SOC-Env/
├── server/
│   ├── app.py                   # FastAPI server exposing /reset, /step, /state
│   ├── environment.py           # Core OpenEnv environment logic
│   ├── grader.py                # Per-task reward and scoring deterministic logic
│   ├── models.py                # Pydantic typed models (Observation, Action, Reward)
│   ├── scenario_loader.py       # Loads and manages scenario JSON files
│   └── scenarios/               # Ground truth & anomaly overlays for tasks
├── inference.py                 # Baseline inference script (LLM bridge)
├── openenv.yaml                 # OpenEnv spec metadata
├── Dockerfile                   # Container definition
└── requirements.txt             # Project dependencies
```

### Execution Flow
The LLM never communicates with the environment directly. `inference.py` acts as the sole bridge between the LLM and the environment server:

```text
inference.py
    │
    ├──► POST /reset?task=task_1    → returns initial org-wide observation
    │
    ├──► LLM via OpenAI API client  → action JSON
    │
    ├──► POST /step (action JSON)   → new observation + reward + done + progress
    │
    └──► repeat until done=true → emit [END] log → continue to next task
```

---

## 2. Environment Details & Observation Space

The observation space returns structured JSON directly conforming to Pydantic models. 
The agent experiences two types of states: `OrgSnapshot` and `DeviceTelemetry`.

**OrgSnapshot (High-Level Telemetry)**: 
Returned on initial reset or `poll_org` action. 
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

**DeviceTelemetry (Deep Drill-down)**: 
Returned when executing `investigate_device`. Includes a detailed breakdown of CPU, memory, active_processes, disk_percent, network behavior (bytes_in/out, ports, flagged IPs), failed logins, outbound IPs, and dns_queries.

---

## 3. Action Space

Agents interact with the environment strictly by emitting valid JSON objects. The following actions define the toolset available to the agent:

| Action | Required Fields | Effect |
|---|---|---|
| `poll_org` | none | Returns fresh org-wide snapshot |
| `investigate_device` | `target_device` | Returns full DeviceTelemetry for one device |
| `isolate_device` | `target_device` | Cuts device from network, irreversible in episode |
| `block_ip` | `target_device`, `target_ip` | Blocks specific IP on device firewall |
| `kill_process` | `target_device`, `target_process` | Terminates named process on device |
| `escalate` | `target_device`, `anomaly_type`, `confidence` | Formally reports incident, closes device investigation |
| `mark_safe` | `target_device` | Declares device clean, closes investigation |

---

## 4. Anomaly Taxonomy

Threats within the environment are grounded into specific anomalies. Identifying them and mapping the correct remediation relies on reading the distinct telemetry markers:

- `data_exfiltration`: High bytes_out, flagged IPs, unknown process
- `malware_process`: Unknown process, high CPU, outbound C2 traffic
- `brute_force_login`: High failed_login_attempts, single source IP
- `suspicious_url_click`: Bad domain in dns_queries, new spawned process
- `port_scan`: High connection count, many dest ports, low data
- `resource_abuse`: CPU/memory pegged, legitimate runaway process

---

## 5. Tasks & Challenges

1. **task_1 (Easy): Single Device Anomaly:** 
   Single device exfiltrating data clearly visible on the org snapshot.
2. **task_2 (Medium): Multi-Device Triage:** 
   6 devices total. 3 produce noisy elevated metrics (false positives), and 1 has a real malware process. The agent must triage efficiently to avoid wasting steps on noisy hardware.
3. **task_3 (Hard): Correlated Multi-Stage Attack:** 
   8 devices total. A workstation clicked a suspicious URL opening a connection to an internal server which is now used to relay exfiltration data. The agent must correlate network flow and active processes across multiple distinct devices to score perfectly.

---

## 6. Inference Workflow (Decision Tree)

The `inference.py` integrates a rigorous set of system instructions that outline the optimal decision tree intended for autonomous execution:

1. **POLL**: Identify suspicious devices from the broad snapshot. 
2. **INVESTIGATE**: Drill down on an unseen flagged device.
3. **REVIEW**: Parse output telemetry properties (`active_processes`, `outbound_ips`, `dns_queries`).
4. **REMEDIATE**: Perform mitigations dynamically (kill processes, block IPs, or isolate completely) based on evidence.
5. **ESCALATE**: Formally report the identified `anomaly_type` alongside its initial remediation to secure maximum efficiency reward limits.
6. **LOOP**: Iterate through remaining anomalies until the complete system is restored or marked safe.

---

## 7. Grader & Reward Function [0.0 - 1.0]

The evaluator uses a deterministic grader that rewards precise identification and remediation while harshly penalizing false positives. The final score is clamped between 0 and 1.

- **Identification Score (max 0.4):** Awarded on escalate. 0.4 requires correct target device + correct anomaly type. Partial score applied for mapping the right device to the wrong anomaly.
- **Remediation Score (max 0.3):** Evaluated at episode end. Agent must utilize perfectly paired remediations (`kill`/`block`/`isolate`) tailored to the specific device threat.
- **Efficiency Score (max 0.2):** Scaled dynamically `min(1.0, optimal_steps / actual_steps) * 0.2`
- **False Positive Penalty (min -0.3):** Applied immediately for executing destructive tasks (isolating/killing/blocking) on healthy, uncompromised devices.
- **Step Penalty:** Applied for every action beyond the mathematically optimal scenario limit.

---

## 8. Setup & Execution Instructions

### Local Environment
We highly recommend using a virtual environment to prevent dependency conflicts.

1. Create and activate a Virtual Environment:
   ```bash
   # On Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # On macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI Environment Server:
   ```bash
   python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
   ```

4. Configure your LLM backend by creating a `.env` file in the root directory. At a minimum, provide your API key. For example, if evaluating with an OpenAI-compatible endpoint:
   ```ini
   # Complete .env example
   OPENAI_API_KEY="your_api_key_here"
   API_BASE_URL="https://router.huggingface.co/v1"
   MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" 
   ```
   *(Note: You can also substitute `OPENAI_API_KEY` for `HF_TOKEN` if authenticating directly to HuggingFace).*

5. Open a **new terminal tab** (with the virtual environment activated), and run the baseline agent:
   ```bash
   python inference.py
   ```

### Docker Usage
If you prefer not to manage local dependencies, the environment supports a fully self-contained Docker release:
```bash
docker build -t soc-env .
docker run -p 7860:7860 soc-env
```
*(After the container starts, strictly run step 4 & 5 locally or run your agent codebase targeting `http://localhost:7860`)*

---

## 9. Baseline Expected Scores

Target expectations for models like `Qwen/Qwen2.5-72B-Instruct`:
- **task_1:** 0.90
- **task_2:** 0.08
- **task_3:** 0.56

*Note: Although task 3 is architecturally more complex than task 2, frontier models currently struggle with task 2's specific multi-device triage due to inherent design flaws in consistent JSON formatting and state tracking.*
