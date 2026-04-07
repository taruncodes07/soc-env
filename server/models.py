from pydantic import BaseModel, ConfigDict
from typing import Literal, List, Optional, Union, Dict

class NetworkDetail(BaseModel):
    bytes_in_mb: float
    bytes_out_mb: float
    active_connections: int
    destination_ports: List[int]
    flagged_ips: List[str]

class DeviceSummary(BaseModel):
    device_id: str
    hostname: str
    status: Literal["healthy", "warning", "critical"]
    cpu_percent: float
    memory_percent: float
    bytes_out_mb: float
    active_connections: int
    alert_flags: List[str]

class OrgSnapshot(BaseModel):
    snapshot_type: Literal["org_wide"] = "org_wide"
    timestamp: str
    devices: List[DeviceSummary]
    step_number: int

class DeviceTelemetry(BaseModel):
    snapshot_type: Literal["device_detail"] = "device_detail"
    device_id: str
    hostname: str
    os: str
    role: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_processes: List[str]
    network: NetworkDetail
    logged_in_user: str
    failed_login_attempts: int
    dns_queries: List[str]
    outbound_ips: List[str] = []
    last_seen: str
    step_number: int

class Observation(BaseModel):
    data: Union[OrgSnapshot, DeviceTelemetry]
    last_action_result: str
    available_actions: List[str]
    episode_done: bool

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
    target_device: Optional[str] = None
    target_ip: Optional[str] = None
    target_process: Optional[str] = None
    anomaly_type: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    identification_score: float
    remediation_score: float
    efficiency_score: float
    false_positive_penalty: float
    final_score: Optional[float] = None

class GroundTruth(BaseModel):
    compromised_devices: List[str]
    anomaly_types: Dict[str, str]
    correct_remediation_actions: Dict[str, List[str]]
    optimal_steps: int
    healthy_devices: List[str]

class ActionRecord(BaseModel):
    step: int
    action: Action
    result: str

class DeviceState(BaseModel):
    telemetry: DeviceTelemetry
    is_compromised: bool
    is_isolated: bool = False
    blocked_ips: List[str] = []
    killed_processes: List[str] = []

class EpisodeState(BaseModel):
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    devices: Dict[str, DeviceState]
    investigated_devices: List[str] = []
    actions_taken: List[ActionRecord] = []
    escalated_devices: List[str] = []
    marked_safe_devices: List[str] = []
    cumulative_reward: float = 0.0
    ground_truth: GroundTruth
    last_action_result: str = "Episode started."
    identification_score: float = 0.0
    remediation_score: float = 0.0
    efficiency_score: float = 0.0
    false_positive_penalty: float = 0.0
    final_score: Optional[float] = None
