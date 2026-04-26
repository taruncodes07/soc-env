# 🛡️ SOC-Env: Security Operations Center RL Environment

**SOC-Env** is an OpenEnv-compliant reinforcement learning environment that simulates a Security Operations Center (SOC) monitoring an organization's device fleet. An AI agent acts as an autonomous SOC analyst — it polls telemetry, triages anomalies, investigates suspicious devices, and executes remediation actions.

This repository is optimized for **stable, high-performance RL training** using a native GRPO (Group Relative Policy Optimization) loop that runs perfectly on consumer hardware (RTX 3050 6GB) and Google Colab.

---

## 🚀 Quick Start (Google Colab)
The fastest way to train and check progress is using our one-shot notebook:
1. Open **`colab_train.ipynb`** in Google Colab.
2. Select **T4 GPU** runtime.
3. Run the "All-in-One" training cell.

---

## 🏛️ Architecture
The environment is built on the **OpenEnv** standard, ensuring deterministic rewards and reproducible triage scenarios.

- **`server/`**: Core environment logic (scenarios, grading, models).
- **`soc_openenv.py`**: OpenEnv-compliant wrapper (inherits from `BaseEnvironment`).
- **`train_rl.py`**: Stable, manual GRPO trainer. No fragile dependencies.
- **`demo_agent.py`**: Clean script to run your trained model and watch it triage live.

---

## 🧠 RL Training Features

### 1. Zero-Dependency GRPO
Unlike fragile black-box trainers, our `train_rl.py` implements the GRPO logic natively using standard `transformers` and `peft`. This avoids library version conflicts and provides 100% stable rewards.

### 2. Multi-Signal Reward Shaping
The agent is guided by three distinct reward signals:
- **Format Reward**: Correct JSON structure.
- **Step Reward**: Efficient navigation through investigation steps.
- **Episode Final Reward**: High-quality signal based on the final triage outcome.

### 3. Progressive Curriculum
Training automatically unlocks harder tasks as the agent improves:
- **Task 1**: Single device exfiltration.
- **Task 2**: Multi-device noise filtering.
- **Task 3**: Correlated multi-stage lateral movement.
- **Task 4**: Advanced honeypot and decoy triage.

### 4. Real-time Monitoring
Track your agent's reasoning via:
- **TensorBoard**: Live loss and reward curves.
- **Matplotlib Plots**: Automatically saved reward graphs in `training_graphs/`.

---

## 🛠️ Local Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Training**:
   ```bash
   python train_rl.py
   ```

3. **Run Demo** (after training):
   ```bash
   python demo_agent.py
   ```

---

## 📊 Anomaly Taxonomy
The agent learns to identify and remediate the following threats based on telemetry markers:
- `data_exfiltration`: High bytes_out, flagged IPs.
- `malware_process`: Unknown process, high CPU.
- `brute_force_login`: High failed_login_attempts.
- `suspicious_url_click`: Malicious DNS domains.
- `resource_abuse`: Legitimate process runaway.

---
*Compliant with the OpenEnv Platform Spec v1.1.0*
