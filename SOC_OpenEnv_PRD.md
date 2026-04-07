# Product Requirements Document
## SOC-Env: Security Operations Center Environment for AI Agents
### OpenEnv Hackathon Round 1 Submission

---

## 1. Project Overview

### 1.1 Summary

SOC-Env is an OpenEnv-compliant reinforcement learning environment that simulates a Security Operations Center (SOC) monitoring an organization's device fleet. An AI agent acts as an autonomous SOC analyst — it polls telemetry, triages anomalies, investigates suspicious devices, and executes remediation actions. It is evaluated and rewarded based on the correctness, precision, and efficiency of its decisions.

### 1.2 Motivation

AI-powered security operations is one of the most commercially relevant and technically unsolved problems in enterprise software. Real SOC analysts deal with alert fatigue, multi-device correlation, and time-pressured triage decisions daily. No OpenEnv environment currently models this domain. SOC-Env fills that gap by providing a deterministic, reproducible, reward-shaped environment where agents can be evaluated and trained on genuine SOC reasoning tasks.

### 1.3 Real-World Task

The environment simulates the following human workflow:
- A SOC analyst monitors a dashboard of all organization devices
- They identify which devices show anomalous signals
- They drill into specific devices for full telemetry
- They correlate signals across devices if needed
- They execute the correct remediation action
- They formally escalate the incident with the correct classification

---

## 2. Architecture Overview

### 2.1 System Components

```
SOC-Env/
├── server/
│   ├── main.py                  # FastAPI server exposing /reset, /step, /state
│   ├── environment.py           # Core OpenEnv environment logic
│   ├── grader.py                # Per-task reward and scoring logic
│   ├── models.py                # Pydantic typed models (Observation, Action, Reward)
│   ├── scenario_loader.py       # Loads and manages scenario JSON files
│   └── scenarios/
│       ├── task_1.json          # Easy: single obvious anomaly
│       ├── task_2.json          # Medium: multi-device, one real threat
│       └── task_3.json          # Hard: correlated multi-stage attack
├── inference.py                 # Baseline inference script (root level, mandatory)
├── openenv.yaml                 # OpenEnv spec metadata
├── Dockerfile                   # Container definition
├── requirements.txt
└── README.md
```

### 2.2 Request Flow

```
inference.py
    │
    ├──► POST /reset?task=task_1    → initial org-wide observation
    │
    ├──► LLM via OpenAI client      → action JSON
    │
    ├──► POST /step (action JSON)   → new observation + reward + done + info
    │
    └──► repeat until done=true → emit [END] log → next task
```

The LLM never communicates with the environment directly. `inference.py` is the sole bridge between the LLM and the environment server.

---

## 3. OpenEnv Specification Compliance

### 3.1 openenv.yaml

```yaml
name: soc-env
version: 1.0.0
description: >
  A Security Operations Center environment where an AI agent monitors
  organization-wide device telemetry, detects anomalies, and executes
  remediation actions.
tasks:
  - id: task_1
    name: Single Device Anomaly
    difficulty: easy
  - id: task_2
    name: Multi-Device Triage
    difficulty: medium
  - id: task_3
    name: Correlated Multi-Stage Attack
    difficulty: hard
action_space: structured_json
observation_space: structured_json
reward_range: [0.0, 1.0]
```

### 3.2 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Initialize episode for given task, return initial observation |
| `/step` | POST | Accept action, return observation + reward + done + info |
| `/state` | GET | Return full current internal state (for debugging/validation) |

### 3.3 Typed Pydantic Models

#### Observation Model
```python
class DeviceSummary(BaseModel):
    device_id: str
    hostname: str
    status: Literal["healthy", "warning", "critical"]
    cpu_percent: float
    memory_percent: float
    bytes_out_mb: float
    active_connections: int
    alert_flags: List[str]  # e.g. ["high_cpu", "unusual_outbound"]

class OrgSnapshot(BaseModel):
    snapshot_type: Literal["org_wide"]
    timestamp: str
    devices: List[DeviceSummary]
    step_number: int

class DeviceTelemetry(BaseModel):
    snapshot_type: Literal["device_detail"]
    device_id: str
    hostname: str
    os: str
    role: str  # workstation | server | router
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_processes: List[str]
    network: NetworkDetail
    logged_in_user: str
    failed_login_attempts: int
    dns_queries: List[str]
    outbound_ips: List[str]
    last_seen: str
    step_number: int

class NetworkDetail(BaseModel):
    bytes_in_mb: float
    bytes_out_mb: float
    active_connections: int
    destination_ports: List[int]
    flagged_ips: List[str]

class Observation(BaseModel):
    data: Union[OrgSnapshot, DeviceTelemetry]
    last_action_result: str        # human-readable result of last action
    available_actions: List[str]   # list of valid action names at this step
    episode_done: bool
```

#### Action Model
```python
class Action(BaseModel):
    action: Literal[
        "poll_org",
        "investigate_device",
        "isolate_device",
        "block_ip",
        "kill_process",
        "escalate",
        "mark_safe"
    ]
    target_device: Optional[str] = None      # device_id
    target_ip: Optional[str] = None          # for block_ip
    target_process: Optional[str] = None     # for kill_process
    anomaly_type: Optional[str] = None       # for escalate
    confidence: Optional[float] = None       # for escalate, 0.0-1.0
    reasoning: Optional[str] = None          # optional, aids debugging
```

#### Reward Model
```python
class Reward(BaseModel):
    step_reward: float              # reward for this specific step
    cumulative_reward: float        # total accumulated so far
    identification_score: float     # 0.0-0.4: correct device + anomaly type
    remediation_score: float        # 0.0-0.3: correct action taken
    efficiency_score: float         # 0.0-0.2: steps used vs optimal
    false_positive_penalty: float   # 0.0 to -0.3: wrong targets penalized
    final_score: Optional[float]    # 0.0-1.0, only set when done=True
```

---

## 4. Telemetry Data Design

### 4.1 Scenario File Structure

Each task is a self-contained JSON file with two top-level sections: `devices` (baseline state for all devices) and `anomaly` (the ground truth overlay and grader metadata).

```json
{
  "scenario_id": "task_1",
  "description": "Single device data exfiltration",
  "devices": [
    {
      "device_id": "device_001",
      "hostname": "WORKSTATION-ALICE",
      "os": "Windows 11",
      "role": "workstation",
      "logged_in_user": "alice",
      "baseline": {
        "cpu_percent": 18.0,
        "memory_percent": 42.0,
        "disk_percent": 55.0,
        "active_processes": ["explorer.exe", "chrome.exe", "outlook.exe"],
        "network": {
          "bytes_in_mb": 2.1,
          "bytes_out_mb": 0.8,
          "active_connections": 4,
          "destination_ports": [443, 80],
          "flagged_ips": []
        },
        "failed_login_attempts": 0,
        "dns_queries": ["google.com", "office365.com"]
      },
      "anomaly_overlay": {
        "cpu_percent": 78.0,
        "network": {
          "bytes_out_mb": 450.0,
          "destination_ports": [443, 80, 6667],
          "flagged_ips": ["185.220.101.45"]
        },
        "active_processes": ["explorer.exe", "chrome.exe", "outlook.exe", "svch0st.exe"],
        "dns_queries": ["google.com", "office365.com", "pastebin.com", "transfer.sh"]
      }
    }
  ],
  "ground_truth": {
    "compromised_device": "device_001",
    "anomaly_type": "data_exfiltration",
    "correct_remediation_actions": ["block_ip", "isolate_device"],
    "optimal_steps": 5,
    "escalate_required": true
  }
}
```

### 4.2 Anomaly Overlay Logic

When the scenario loads, the environment merges `anomaly_overlay` fields onto the baseline for the compromised device only. All other devices retain their pure baseline values. The agent never sees the ground truth section — it only exists inside the grader.

The org-wide snapshot shows elevated `status` and `alert_flags` for the compromised device derived from the merged state, giving the agent enough signal to identify which device warrants investigation without giving away the full detail.

### 4.3 Anomaly Taxonomy

| Anomaly Type | Primary Telemetry Signals | Severity | Correct Remediation |
|---|---|---|---|
| `data_exfiltration` | High bytes_out, flagged IPs, unknown process | Critical | `block_ip` → `isolate_device` |
| `malware_process` | Unknown process, high CPU, outbound C2 traffic | Critical | `kill_process` → `isolate_device` |
| `brute_force_login` | High failed_login_attempts, single source IP | High | `block_ip` → `escalate` |
| `suspicious_url_click` | Bad domain in dns_queries, new spawned process | Medium | `kill_process` → `escalate` |
| `port_scan` | High connection count, many dest ports, low data | Medium | `block_ip` → `escalate` |
| `resource_abuse` | CPU/memory pegged, legitimate runaway process | Low | `kill_process` → `mark_safe` |

---

## 5. Action Space Definition

### 5.1 Valid Actions

| Action | Required Fields | Effect | Step Cost |
|---|---|---|---|
| `poll_org` | none | Returns fresh org-wide snapshot | 1 step |
| `investigate_device` | `target_device` | Returns full DeviceTelemetry for one device | 1 step |
| `isolate_device` | `target_device` | Cuts device from network, irreversible in episode | 1 step |
| `block_ip` | `target_device`, `target_ip` | Blocks specific IP on device firewall | 1 step |
| `kill_process` | `target_device`, `target_process` | Terminates named process on device | 1 step |
| `escalate` | `target_device`, `anomaly_type`, `confidence` | Formally reports incident, closes device investigation | 1 step |
| `mark_safe` | `target_device` | Declares device clean, closes investigation | 1 step |

### 5.2 Action Validation Rules

- `investigate_device`, `isolate_device`, `block_ip`, `kill_process`, `escalate`, `mark_safe` all require a valid `target_device` that exists in the current scenario
- `block_ip` requires `target_ip` to be a string in valid IP format
- `kill_process` requires `target_process` to match a process name visible in the device's telemetry
- `escalate` requires `anomaly_type` to be one of the 6 defined anomaly types
- Invalid actions return an error in `last_action_result` and incur a small step penalty but do not terminate the episode

### 5.3 LLM Response Format

The LLM must respond with a JSON object only — no prose, no markdown fences. The inference script parses this directly into the Action model.

```json
{
  "action": "investigate_device",
  "target_device": "device_003",
  "reasoning": "device_003 shows critical status with high bytes_out and flagged IPs in org snapshot"
}
```

```json
{
  "action": "escalate",
  "target_device": "device_003",
  "anomaly_type": "data_exfiltration",
  "confidence": 0.92,
  "reasoning": "450MB outbound to flagged IP 185.220.101.45 via unknown process svch0st.exe"
}
```

---

## 6. Task Definitions

### 6.1 Task 1 — Easy: Single Device Anomaly

**Scenario:** One device in a 4-device org is exfiltrating data. The org-wide snapshot makes the anomaly visually obvious — one device is `critical` with high bytes_out and flagged IPs. All other devices are `healthy` with completely normal metrics.

**Optimal Agent Behavior:**
1. Receive org snapshot (auto on reset)
2. `investigate_device(device_001)` — confirms data exfiltration signals
3. `block_ip(device_001, "185.220.101.45")` — stops exfiltration
4. `isolate_device(device_001)` — contains the breach
5. `escalate(device_001, "data_exfiltration", 0.95)` — formal report

**Optimal steps:** 5
**Max steps:** 12
**Expected score for capable LLM:** 0.85–1.0
**Anomaly injected:** `data_exfiltration`

---

### 6.2 Task 2 — Medium: Multi-Device Triage

**Scenario:** 6 devices in the org. 3 devices show slightly elevated metrics that are within normal variance (noisy but healthy). 1 device has a real `malware_process` anomaly. The agent must correctly triage — investigating the 3 noisy devices wastes steps and reduces efficiency score.

**Optimal Agent Behavior:**
1. Receive org snapshot
2. Identify device with `critical` status vs devices with `warning` (noise)
3. `investigate_device(device_004)` — confirms unknown process and C2 traffic
4. `kill_process(device_004, "svch0st.exe")`
5. `isolate_device(device_004)`
6. `escalate(device_004, "malware_process", 0.90)`

**Optimal steps:** 6
**Max steps:** 15
**Expected score for capable LLM:** 0.60–0.80
**Anomaly injected:** `malware_process`
**Triage trap:** Devices 2, 3, 5 show `warning` status due to natural load — investigating any of them costs steps with no payoff

---

### 6.3 Task 3 — Hard: Correlated Multi-Stage Attack

**Scenario:** 8-device org. A workstation (`device_002`) clicked a malicious URL 3 steps ago — dns_queries shows the bad domain and a new process was spawned. That process has since established a connection to a server device (`device_006`) which is now acting as a relay and exfiltrating data. Neither device alone tells the full story — `device_002` looks medium-severity, `device_006` looks like a busy server. The agent must correlate the two.

**Optimal Agent Behavior:**
1. Receive org snapshot
2. `investigate_device(device_002)` — sees suspicious DNS + spawned process
3. `investigate_device(device_006)` — sees unusual outbound to same flagged IP range
4. Correlate: same flagged IP family, process on device_002 opened connection to device_006
5. `kill_process(device_002, "payload.exe")`
6. `block_ip(device_006, "185.220.101.45")`
7. `isolate_device(device_002)`
8. `isolate_device(device_006)`
9. `escalate(device_002, "suspicious_url_click", 0.88)`
10. `escalate(device_006, "data_exfiltration", 0.88)`

**Optimal steps:** 10
**Max steps:** 20
**Expected score for capable LLM:** 0.30–0.55
**Anomaly injected:** `suspicious_url_click` on device_002, `data_exfiltration` relay on device_006
**Challenge:** No single device reveals the full attack chain. Requires cross-device reasoning.

---

## 7. Reward Function

### 7.1 Component Breakdown

All components sum to produce the final score, clamped to [0.0, 1.0].

#### Identification Reward (max 0.4)
Awarded when agent calls `escalate()`:
- Correct device AND correct anomaly_type → **+0.4**
- Correct device, wrong anomaly_type → **+0.2**
- Wrong device, correct anomaly_type → **+0.1**
- Wrong device, wrong anomaly_type → **+0.0**

For Task 3 (two devices): identification reward is split 0.2 per device, same partial credit logic applies per device.

#### Remediation Reward (max 0.3)
Evaluated at episode end against `correct_remediation_actions` in ground truth:
- All correct remediation actions executed on correct device → **+0.3**
- Half of required actions correct → **+0.15**
- Correct actions on wrong device → **+0.0**
- No remediation actions taken → **+0.0**

#### Efficiency Reward (max 0.2)
Calculated as:
```
efficiency = min(1.0, optimal_steps / actual_steps)
efficiency_reward = efficiency * 0.2
```
An agent that takes exactly optimal steps scores 0.2. An agent that takes twice as many steps scores 0.1.

#### False Positive Penalty (min -0.3)
Applied immediately when the action occurs:
- `isolate_device` on a healthy device → **-0.15** per incident
- `kill_process` on a healthy device → **-0.10** per incident
- `block_ip` on a healthy device → **-0.05** per incident

#### Step Penalty
Applied every step beyond optimal:
- **-0.01** per step beyond `optimal_steps`
- Creates continuous pressure, prevents idle looping

### 7.2 Anti-Exploit Design

A blind agent that immediately calls `escalate(device_001, "data_exfiltration")` without any investigation cannot score high because:
- It has not called any remediation actions → remediation_score = 0.0
- It has not called `investigate_device` → it cannot know the anomaly type from org snapshot alone
- If it guesses wrong anomaly type → identification_score = 0.1 max
- Maximum possible blind score ≈ 0.1 (lucky guess on type) + 0.2 (efficiency) = 0.3

A proper investigation and remediation scores 0.85–1.0. The reward function is exploit-resistant.

### 7.3 Reward Signal Timing

| Event | When Reward Is Emitted |
|---|---|
| Invalid action | Immediately, -0.02 |
| False positive destructive action | Immediately |
| Unnecessary `poll_org` | After 1st free poll, -0.01 per additional |
| `escalate` called | Identification score computed immediately |
| Episode ends (done=True) | Remediation + efficiency scores added, final_score set |

---

## 8. Environment State Management

### 8.1 State Schema

```python
class EpisodeState(BaseModel):
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    devices: Dict[str, DeviceState]          # full internal device states
    investigated_devices: List[str]           # devices agent has drilled into
    actions_taken: List[ActionRecord]         # full action history
    escalated_devices: List[str]             # devices formally escalated
    marked_safe_devices: List[str]           # devices marked clean
    cumulative_reward: float
    ground_truth: GroundTruth                # never exposed to agent
```

### 8.2 Episode Lifecycle

- `reset(task_id)` → loads scenario JSON, initializes EpisodeState, returns initial OrgSnapshot
- Each `step(action)` → validates action, updates state, computes step reward, checks done condition, returns observation + reward + done + info
- `done=True` is triggered by: correct escalation of all compromised devices, agent calls `mark_safe` on all remaining devices, or `step_number >= max_steps`
- `state()` → returns full EpisodeState for debugging (ground_truth included, used by validator)

### 8.3 Clean Reset Guarantee

On every `reset()` call, all episode state is fully replaced. No state leaks between tasks. The server is stateless across episodes by design — the scenario JSON is the only source of truth.

---

## 9. Inference Script Specification

### 9.1 File: `inference.py` (root level)

The script must run all 3 tasks sequentially and emit compliant stdout logs.

### 9.2 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | none | API key for LLM calls |
| `SOC_ENV_URL` | `http://localhost:7860` | Environment server URL |

### 9.3 LLM System Prompt

```
You are an AI SOC analyst. You monitor an organization's device fleet for security anomalies.

Each step you receive either:
1. An org-wide snapshot showing all devices with high-level status
2. A detailed telemetry report for a specific device you investigated

You must respond with ONLY a valid JSON action object. No prose, no markdown, no explanation outside the JSON.

Valid actions:
- {"action": "poll_org"} — refresh the org-wide snapshot
- {"action": "investigate_device", "target_device": "<id>"} — get full telemetry for a device
- {"action": "isolate_device", "target_device": "<id>"} — cut device from network (irreversible)
- {"action": "block_ip", "target_device": "<id>", "target_ip": "<ip>"} — block an IP on a device
- {"action": "kill_process", "target_device": "<id>", "target_process": "<name>"} — terminate process
- {"action": "escalate", "target_device": "<id>", "anomaly_type": "<type>", "confidence": 0.0-1.0} — formal incident report
- {"action": "mark_safe", "target_device": "<id>"} — declare device clean

Anomaly types: data_exfiltration | malware_process | brute_force_login | suspicious_url_click | port_scan | resource_abuse

Rules:
- Only take destructive actions (isolate, kill, block) if you have evidence from investigating the device
- Always escalate after remediating a compromised device
- Be efficient — unnecessary steps reduce your score
- Include optional "reasoning" field to explain your decision
```

### 9.4 Stdout Format (Mandatory)

```
[START] task=task_1 env=soc-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=investigate_device reward=0.00 done=false error=null
[STEP] step=2 action=block_ip reward=0.05 done=false error=null
[STEP] step=3 action=isolate_device reward=0.10 done=false error=null
[STEP] step=4 action=escalate reward=0.75 done=true error=null
[END] success=true steps=4 score=0.88 rewards=0.00,0.05,0.10,0.75
```

### 9.5 Max Steps Per Task

| Task | Max Steps |
|---|---|
| task_1 | 12 |
| task_2 | 15 |
| task_3 | 20 |

Total worst-case runtime: 47 LLM calls × ~4s average = ~3 minutes. Well within 20-minute limit.

---

## 10. Grader Implementation

### 10.1 Grader Class Interface

```python
class TaskGrader:
    def __init__(self, ground_truth: GroundTruth): ...
    def score_step(self, action: Action, state: EpisodeState) -> float: ...
    def score_episode(self, state: EpisodeState) -> Reward: ...
    def is_done(self, state: EpisodeState) -> bool: ...
```

### 10.2 Grader Ground Truth Fields

```python
class GroundTruth(BaseModel):
    compromised_devices: List[str]                    # list of device_ids
    anomaly_types: Dict[str, str]                     # device_id -> anomaly_type
    correct_remediation_actions: Dict[str, List[str]] # device_id -> [action names]
    optimal_steps: int
    healthy_devices: List[str]                        # all non-compromised device_ids
```

### 10.3 Determinism Guarantee

All grader logic is purely deterministic — no randomness, no LLM calls inside the grader. Given the same action sequence, the grader always produces the same score. This ensures reproducibility across judge re-runs.

---

## 11. Deployment

### 11.1 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
COPY openenv.yaml .
EXPOSE 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 11.2 Requirements

```
fastapi
uvicorn
pydantic>=2.0
openai
python-dotenv
```

### 11.3 HuggingFace Space Configuration

- Space SDK: Docker
- Tags: `openenv`
- Port: 7860
- The FastAPI server runs directly — no additional HF-specific configuration needed

### 11.4 Resource Constraints

The environment is entirely in-memory. No database, no external API calls from the server. Scenario JSON files are loaded at reset time. This keeps the server lightweight and ensures it runs cleanly on vcpu=2, memory=8GB as required.

---

## 12. README Requirements

The README must include:

1. **Environment description** — what SOC-Env simulates and why it matters for the agent research community
2. **Observation space** — description of OrgSnapshot and DeviceTelemetry formats with field explanations
3. **Action space** — all 7 actions with required fields and effects
4. **Task descriptions** — all 3 tasks with difficulty, scenario description, and expected agent behavior
5. **Reward function** — breakdown of all 5 components with exact weights
6. **Setup instructions** — local dev (uvicorn), Docker, and HuggingFace Space usage
7. **Baseline scores** — reproducible scores from the baseline inference run on all 3 tasks
8. **Anomaly type reference** — all 6 anomaly types with description and telemetry signatures

---

## 13. Baseline Expected Scores

These are the target scores the baseline LLM (Qwen 72B) should achieve. The grader and scenario design should be validated against these targets before submission:

| Task | Expected Score | Notes |
|---|---|---|
| task_1 | 0.75 – 0.90 | Straightforward for a capable model |
| task_2 | 0.50 – 0.70 | Triage noise will cost some efficiency |
| task_3 | 0.25 – 0.50 | Multi-device correlation is genuinely hard |

If task_1 scores below 0.6 consistently, the scenario or system prompt needs adjustment. If task_3 scores above 0.7 consistently, the scenario is too easy and the correlation challenge needs strengthening.

---

## 14. Evaluation Self-Check

Before submission, verify all of the following:

- [ ] `/reset` returns HTTP 200 with valid Observation JSON
- [ ] `/step` with valid action returns Observation + Reward + done + info
- [ ] `/state` returns full EpisodeState
- [ ] `openenv validate` passes with no errors
- [ ] `docker build` completes with no errors
- [ ] `docker run` starts and server responds to `/reset`
- [ ] `inference.py` runs end-to-end without exceptions
- [ ] All 3 tasks produce `[START]`, `[STEP]*`, `[END]` logs
- [ ] All scores in `[END]` are between 0.0 and 1.0
- [ ] Scores are reproducible across two runs (same scenario, same seed)
- [ ] A blind agent that skips investigation cannot score above 0.35
- [ ] Grader returns different scores for good vs bad action sequences
- [ ] `openenv.yaml` contains all required fields
- [ ] HF Space URL responds to `/reset` with HTTP 200

---

*PRD Version 1.0 — SOC-Env OpenEnv Hackathon Round 1*
