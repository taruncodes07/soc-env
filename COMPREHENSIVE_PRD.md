# Comprehensive Product Requirements Document (PRD) for SOC-Env

## 1. Executive Summary & Core Mission
SOC-Env is an OpenEnv-compliant reinforcement learning environment simulating a Security Operations Center (SOC). It challenges AI agents to act as autonomous SOC analysts. The core workflow requires the agent to poll for telemetry, investigate anomalies across an organization's device fleet, correlate multi-device events, remediate specific threats, and escalate incidents without unnecessarily interrupting healthy devices (avoiding false positives).

To ensure exact code regeneration, this detailed PRD unifies all discrete structures, models, algorithms, scenarios, pipeline configurations, and agent inference routines from the project repository. Code-by-code translation rules are meticulously mapped.

---

## 2. Directory Structure
```text
SOC-Env/
├── .env                              # API Configurations
├── server/
│   ├── app.py                   # FastAPI server endpoints
│   ├── environment.py           # Core step/reset environment loop
│   ├── grader.py                # Deterministic scoring functions
│   ├── models.py                # Pydantic schemas (Observation, Action)
│   ├── scenario_loader.py       # JSON anomaly overlay logic
│   └── scenarios/               # Ground truth overlays
│       ├── task_1.json
│       ├── task_2.json
│       └── task_3.json
├── inference.py                 # LLM bridge evaluation script
├── openenv.yaml                 # OpenEnv metadata
├── pyproject.toml               # Python build config
├── requirements.txt             # Dependency list
├── Dockerfile                   # HF Space/Containerization setup
└── README.md                    # Setup & workflow guide
```

---

## 3. Metadata & Dependency Specifications

### 3.1 `openenv.yaml`
Declares the OpenEnv benchmark specifications.
* **Name**: soc-env
* **Version**: 1.0.0
* **Tasks**:
  * `task_1` - Single Device Anomaly (Difficulty: easy)
  * `task_2` - Multi-Device Triage (Difficulty: medium)
  * `task_3` - Correlated Multi-Stage Attack (Difficulty: hard)
* **Action Space**: `structured_json`
* **Observation Space**: `structured_json`
* **Reward Range**: `[0.0, 1.0]`

### 3.2 Build & Environment Definitions
* **`requirements.txt`**: Needs `fastapi`, `uvicorn[standard]`, `pydantic>=2.0`, `openai`, `python-dotenv`, `requests`, `openenv-core>=0.2.0`.
* **`pyproject.toml`**: Uses `setuptools>=61.0`. Defines script entrypoint `server = "server.app:main"`.
* **`.env`**: Must define at least `API_BASE_URL="https://router.huggingface.co/v1"`, `MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"`, and `HF_TOKEN`. Optional fallback to `OPENAI_API_KEY`.
* **`Dockerfile`**: Requires `python:3.11-slim`, port `7860`, creates non-root user `user` with UID `1000`, copies files with `--chown=user`, and runs `CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]`.

---

## 4. Full Data Models Schema (`server/models.py`)
All environment boundaries exchange data strictly through typed Pydantic models.

### Telemetry Sub-objects
* **`NetworkDetail`**: contains `bytes_in_mb` (float), `bytes_out_mb` (float), `active_connections` (int), `destination_ports` (List[int]), and `flagged_ips` (List[str]).

### Observation Structures
* **`DeviceSummary`**: `device_id`, `hostname`, `status` (Literal healthy/warning/critical), `cpu_percent`, `memory_percent`, `bytes_out_mb`, `active_connections`, `alert_flags` (List[str]).
* **`OrgSnapshot`**: `snapshot_type` (Literal "org_wide"), `timestamp` (str), `devices` (List[DeviceSummary]), `step_number` (int).
* **`DeviceTelemetry`**: `snapshot_type` (Literal "device_detail"), `device_id`, `hostname`, `os`, `role`, `cpu_percent`, `memory_percent`, `disk_percent`, `active_processes` (List[str]), `network` (NetworkDetail), `logged_in_user`, `failed_login_attempts`, `dns_queries` (List[str]), `outbound_ips` (List[str] = []), `last_seen` (str), `step_number` (int).

### Core Return Types
* **`EpisodeProgress`**: Exposes `step_number`, `max_steps`, `investigated_devices`, `escalated_devices`, `marked_safe_devices` to the agent.
* **`Observation`**: The master API output. Combines `data` (Union[OrgSnapshot, DeviceTelemetry]), `last_action_result` (str), `available_actions` (List[str]), `episode_done` (bool), and optional `progress` (EpisodeProgress).
* **`Action`**: Takes `action` parameter which must be one of `[poll_org, investigate_device, isolate_device, block_ip, kill_process, escalate, mark_safe]`. Optional fields based on action: `target_device`, `target_ip`, `target_process`, `anomaly_type`, `confidence`, `reasoning`.
* **`Reward`**: Holds calculation splits: `step_reward`, `cumulative_reward`, `identification_score`, `remediation_score`, `efficiency_score`, `false_positive_penalty`, and `final_score`.

### State & Tracking
* **`GroundTruth`**: The master hidden state containing `compromised_devices`, `anomaly_types` (Dict), `correct_remediation_actions` (Dict), `optimal_steps`, and `healthy_devices`.
* **`ActionRecord`**: History object recording `step`, `action`, & `result`.
* **`DeviceState`**: Internal tracking element binding `telemetry` + true boolean tags: `is_compromised`, `is_isolated`, arrays for `blocked_ips` and `killed_processes`.
* **`EpisodeState`**: Master class tracking `task_id`, `step_number`, `max_steps`, `done`, `devices`, all state arrays (`investigated_devices`, actions taken, escalation sets, safe marks), scores, and the nested `ground_truth`. Default strings initialized (e.g., `last_action_result = "Episode started."`).

---

## 5. REST API Layer (`server/app.py`)
Served via FastAPI without DBs. Fully stateless across resets. Core endpoints:
1. `GET /`: Redirects to `/docs`.
2. `POST /reset?task=<task_id>`: Invokes `env.reset(task_id)`, returning an `Observation`. Captures and raises 400 for bad tasks.
3. `POST /step`: Takes `Action` payload. Requires current episode to not be `done`. Yields `StepResponse` containing `observation`, `reward` (dict format), `done` bool, and `info` dict.
4. `GET /state`: Retrieves full raw `EpisodeState` for evaluation grading visibility.

---

## 6. Pre-Computed Scenario Loading (`server/scenario_loader.py`)
Implements the deep-merge dictionary generation engine. Rather than hand-crafting full payloads, scenarios have a **baseline**, and compromised devices merge a partial **anomaly_overlay**.

* **`load_scenario(task_id)`**: Opens equivalent JSON payload.
* **Merge Rule**: Recursively updates complex types. Arrays are entirely overwritten by the overlay, and dicts are deep-updated.
* **Ground Truth Expansion**: Translates comma-separated strings for `compromised_device`, `anomaly_type`, and `correct_remediation_actions` dynamically into list and dict mappings to populate the `GroundTruth` Pydantic model.
* **Network mapping**: Extracts `flagged_ips` from the network object directly into the top-level `outbound_ips` on `DeviceTelemetry`. Applies ISO timestamps to `last_seen`.

---

## 7. The Three Baseline Scenarios

### Task 1: Single Device Data Exfiltration
* **Topology**: 4 Devices (Workstation-Alice, Workstation-Bob, Server-App-1, Router-Main).
* **Compromised**: `device_001` (Alice).
* **Anomaly**: `data_exfiltration`.
* **Overlay**: CPU pushed to 78%, `svch0st.exe` injected, `bytes_out_mb` pegged at 450MB, flagged IP `185.220.101.45`, malicious DNS included `transfer.sh`.
* **Ground Truth Required Actions**: `block_ip`, `isolate_device`. Optimal Steps: 5. Max steps: 12.

### Task 2: Multi-Device Triage
* **Topology**: 6 Devices. Devices 2, 3, 5 are heavy-use baselines configured specifically to trigger the `warning` status organically (`cpu_percent` > 70%).
* **Compromised**: `device_004`.
* **Anomaly**: `malware_process`.
* **Overlay**: CPU 95%, `svch0st.exe` injected, outbound port 8080 active, flagged IPs `192.168.100.5`, DNS queries to `c2-server.evil.com`.
* **Ground Truth Required Actions**: `kill_process`, `isolate_device`. Optimal Steps: 6. Max steps: 15.

### Task 3: Correlated Multi-Stage Attack
* **Topology**: 8 Devices.
* **Compromised Multi-Point**: `device_002` (Beta Workstation) & `device_006` (Data Server XY).
* **Correlated Anomaly**: `device_002` = `suspicious_url_click`, `device_006` = `data_exfiltration`.
* **Evidence**: Device 2 has DNS `suspicious-domain.ly` and process `payload.exe` connecting to port 4444. Device 6 has massive `bytes_out` (1800MB) mapping to a flagged IP globally on matching port 4444.
* **Ground Truth Required Actions**:
  * Device 2: `kill_process`, `isolate_device`.
  * Device 6: `block_ip`, `isolate_device`.
* Optimal Steps: 10. Max steps: 20.

---

## 8. Environment Step Engine (`server/environment.py`)

### Initial Overview Array Generation (`_get_org_snapshot`)
Distills telemetry into summarized tags:
* **"high_cpu"**: Granted if cpu_percent > 80.
* **"unusual_outbound"**: Granted if bytes_out_mb > 100.
* **"login_failures"**: Granted if failed logic > 0.
* **"flagged_ips"**: Granted if length of list is > 0.
* **Status Enum Mapping**:
  * Critical: >= 2 flags OR CPU > 90.
  * Warning: exactly 1 flag OR CPU > 70.
  * Healthy: 0 flags and CPU <= 70.

### Action Validation Logic Execution
* `poll_org`: Unnecessary polls after step 1 dock -0.01 immediately. Submits org overview.
* `investigate_device`: Tracks device in progress sets. Re-sets view logic to `obs_data = dev_state.telemetry`.
* `isolate_device`: Disconnects server network virtually. Flags boolean flag `dev_state.is_isolated = True`.
* `block_ip`: Requires valid target_ip parameter. Pushes string IP to `blocked_ips`.
* `kill_process`: Requires string param target_process. Pushes string to `dev_state.killed_processes`.
* `mark_safe`: Flags device explicitly to safe list logically required for finishing cleanly.
* `escalate`: Immediately applies evaluation. Assigns fractions of max 0.4 based on target exactness vs target device. Updates lists.
**Validation Check**: Any malformed property (`None` target device etc) produces `Error:` prefix locally pushing `-0.02` direct penalty.

---

## 9. The Deterministic Grader Algorithm (`server/grader.py`)

Scoring is bounded between `[0.01, 0.99]` clamping cleanly to OpenEnv boundaries.

### False Positive Application
Applied decrementally off intermediate steps, aggregated as `state.false_positive_penalty`:
* `isolate_device` against healthy device: `-0.15`
* `kill_process` against healthy device: `-0.10`
* `block_ip` against healthy device: `-0.05`
*(Healthy interactions correctly `mark_safe` afford an additive `+0.05` reward).*

### Step Penalties
Beyond `optimal_steps` defined by Ground Truth, all step calls deduct `-0.01` linearly.

### Completion Check (`is_done`)
`True` if:
1. Agent hits `max_steps`.
2. Compromised elements handled AND (all investigated devices have explicitly been tagged `escalated` or `marked_safe` respectively).

### Final Episode Rollup Score Algorithm
1. **Identification Max(0.4)**:
   * Computed live during `escalate` action per compromised baseline point: Correct Target & Anomaly Type == Max score. Divides by factor of 2 for partial matches.
2. **Remediation Max(0.3)**:
   * Checked iteratively upon episode close matching `req_actions` vs exactly pushed action records. `correct_count == total required -> +Full Slice`. Half completions yield half partials.
3. **Efficiency Score Max(0.2)**:
   * Computed via min operator against `1.0`: `min(1.0, optimal_steps / actual_steps) * 0.2`.
4. **Final Formulation**:
   * sum of (Identification + Remediation + Efficiency) - (False Positive penalties subtracted off) clamped max.

---

## 10. Agent Baseline Inference (`inference.py`)

Script represents a standalone client managing interactions between the environment API (`http://localhost:7860`) and the LLM endpoint.

### System Prompt Logic Tree
Injected exactly to constrain decision spaces securely:
* **Workflow Directive Prompt**:
  * 1. **POLL**
  * 2. **INVESTIGATE** (Checks arrays to prevent redundant loops).
  * 3. **REVIEW** (Compare metrics, looking for anomalies).
  * 4. **REMEDIATE** (Match mitigation to condition).
  * 5. **ESCALATE** (Closes instance).
  * 6. **LOOP** (Repeats until completely swept).

### The Feedback Context Loop
Applies a lean context loop approach to stay under token limits (200 generated tokens max per step):
Includes dynamically concatenated `progress_block` stating exact steps (`Step: X/Y`), followed by `history_block` of locally stored concatenated action strings, appending lastly `CURRENT STATE` from observation API.

### Fallback Tolerances
Implements local text block scrubbing parsing strings ````json` if model drifts formatting. Returns basic `poll_org` if complete syntax failure occurs internally to avoid trace backs.

Outputs rigid standard logging syntax ensuring compatible benchmark runner collection:
`[START] task={task} env={env} model={model}`
`[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error}`
`[END] success={bool} steps={steps} score={score:.2f} rewards={array}`
