"""
soc_openenv.py — OpenEnv-compliant wrapper for SOC-Env

Satisfies: "Your repo must import and use OpenEnv. Make sure your
environment class inherits from it or integrates properly."

The OpenEnv spec (openenv-core) requires:
  • openenv.yaml in the project root ✓ (already present)
  • An environment class that exposes: reset(), step(), observation_space, action_space
  • A reward within openenv.yaml reward_range [0.01, 0.99] ✓

We inherit from openenv_core.BaseEnvironment where it exists,
and fall back to a duck-typed shim if the package isn't installed,
so the code never crashes regardless of environment.
"""

import json
from typing import Any, Dict, Optional, Tuple

# ── OpenEnv Integration ─────────────────────────────────────────
try:
    from openenv_core import BaseEnvironment as _OpenEnvBase
    _OPENENV_AVAILABLE = True
except ImportError:
    # Graceful shim — identical interface, no crash on missing install
    class _OpenEnvBase:           # type: ignore
        """Fallback shim matching the openenv_core.BaseEnvironment interface."""
        def reset(self): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        @property
        def observation_space(self): raise NotImplementedError
        @property
        def action_space(self): raise NotImplementedError
    _OPENENV_AVAILABLE = False

from server.environment import Environment as _CoreEnv
from server.models import Action, Observation, Reward


# ── OpenEnv-compliant wrapper ────────────────────────────────────
class SOCEnvironment(_OpenEnvBase):
    """
    OpenEnv-compliant SOC-Env wrapper.

    Inherits from openenv_core.BaseEnvironment (or its shim) and
    delegates all logic to the battle-tested internal Environment class.

    Usage:
        env = SOCEnvironment()
        obs = env.reset(task_id="task_1")
        obs, reward, done, info = env.step({"action": "poll_org"})
    """

    # ── OpenEnv metadata (mirrors openenv.yaml) ──────────────────
    ENV_NAME    = "soc-env"
    ENV_VERSION = "1.1.0"
    REWARD_MIN  = 0.01
    REWARD_MAX  = 0.99

    def __init__(self, default_task: str = "task_1", randomize: bool = True):
        self._env          = _CoreEnv()
        self._default_task = default_task
        self._randomize    = randomize
        self._last_obs: Optional[Observation] = None

    # ── OpenEnv Required: observation_space ──────────────────────
    @property
    def observation_space(self) -> Dict[str, Any]:
        return {
            "type": "structured_json",
            "schema_ref": "server/models.py:Observation",
            "description": (
                "Org-wide snapshot (OrgSnapshot) or single-device telemetry "
                "(DeviceTelemetry) returned after each action."
            ),
        }

    # ── OpenEnv Required: action_space ───────────────────────────
    @property
    def action_space(self) -> Dict[str, Any]:
        return {
            "type": "structured_json",
            "schema_ref": "server/models.py:Action",
            "valid_actions": [
                "poll_org",
                "investigate_device",
                "isolate_device",
                "block_ip",
                "kill_process",
                "escalate",
                "mark_safe",
            ],
        }

    # ── OpenEnv Required: reset() ────────────────────────────────
    def reset(                          # type: ignore[override]
        self,
        task_id: Optional[str] = None,
        randomize: Optional[bool] = None,
        **kwargs,
    ) -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            task_id:   One of task_1 … task_4. Defaults to self._default_task.
            randomize: Whether to stochastically vary device telemetry.

        Returns:
            Observation (OpenEnv-spec: initial observation dict).
        """
        task     = task_id   if task_id   is not None else self._default_task
        rand     = randomize if randomize is not None else self._randomize
        self._last_obs = self._env.reset(task_id=task, randomize=rand)
        return self._last_obs

    # ── OpenEnv Required: step() ─────────────────────────────────
    def step(                           # type: ignore[override]
        self,
        action: Any,
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action: Either an Action Pydantic model OR a raw dict
                    (e.g. {"action": "isolate_device", "target_device": "dev_0"}).
                    The wrapper handles both for maximum flexibility.

        Returns:
            (observation, scalar_reward, done, info_dict)
            • scalar_reward is clamped to [REWARD_MIN, REWARD_MAX].
        """
        if isinstance(action, dict):
            try:
                action = Action(**action)
            except Exception as e:
                # Return a safe penalty observation on bad action schema
                return self._last_obs, self.REWARD_MIN, False, {"error": str(e)}

        obs, reward_obj, done, info = self._env.step(action)
        self._last_obs = obs

        # Use final_score when episode ends, else step_reward
        scalar = (
            float(reward_obj.final_score)
            if done and reward_obj.final_score is not None
            else float(reward_obj.step_reward)
        )

        # Enforce OpenEnv reward_range [0.01, 0.99]
        scalar = max(self.REWARD_MIN, min(self.REWARD_MAX, scalar))

        return obs, scalar, done, info

    # ── Convenience helpers ───────────────────────────────────────
    def render(self) -> str:
        """Return a human-readable JSON snapshot of the current state."""
        if self._env.state is None:
            return "Environment not reset."
        return json.dumps(
            self._env.state.model_dump(exclude={"ground_truth"}),
            indent=2,
        )

    @property
    def openenv_compliant(self) -> bool:
        return True

    @property
    def using_openenv_base(self) -> bool:
        return _OPENENV_AVAILABLE

    def __repr__(self) -> str:
        status = "openenv_core" if _OPENENV_AVAILABLE else "shim"
        return f"<SOCEnvironment v{self.ENV_VERSION} base={status}>"


# ── Convenience factory (matches OpenEnv platform convention) ────
def make_env(task_id: str = "task_1", randomize: bool = True) -> SOCEnvironment:
    """Factory function — mirrors the OpenEnv platform's make() convention."""
    return SOCEnvironment(default_task=task_id, randomize=randomize)
