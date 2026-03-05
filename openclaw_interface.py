import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class OpenClawBridge:
    """Lightweight HTTP bridge for synchronizing SciOracle state with an OpenClaw-compatible service."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.agent_id = str(config.get("agent_id", "scioracle"))
        self.timeout_s = float(config.get("timeout_s", 5))
        self.auth_token = config.get("auth_token") or os.getenv("OPENCLAW_TOKEN", "")
        self.push_path = config.get("push_state_path", "/api/v1/agents/{agent_id}/state")
        self.pull_path = config.get("pull_commands_path", "/api/v1/agents/{agent_id}/commands")
        self.heartbeat_path = config.get("heartbeat_path", "/api/v1/agents/{agent_id}/heartbeat")

    def is_active(self) -> bool:
        return self.enabled and bool(self.base_url)

    def _build_url(self, template: str) -> str:
        path = template.format(agent_id=urllib.parse.quote(self.agent_id, safe=""))
        return f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _post(self, path_template: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_active():
            return {"ok": False, "error": "bridge_disabled"}

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=self._build_url(path_template),
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8").strip()
                return json.loads(raw) if raw else {"ok": True}
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="ignore")
            return {"ok": False, "error": f"http_{err.code}", "detail": detail}
        except Exception as err:
            return {"ok": False, "error": str(err)}

    def _get(self, path_template: str) -> Dict[str, Any]:
        if not self.is_active():
            return {"ok": False, "error": "bridge_disabled", "commands": []}

        request = urllib.request.Request(
            url=self._build_url(path_template),
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8").strip()
                payload = json.loads(raw) if raw else {}
                if "commands" not in payload:
                    payload["commands"] = []
                return payload
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="ignore")
            return {"ok": False, "error": f"http_{err.code}", "detail": detail, "commands": []}
        except Exception as err:
            return {"ok": False, "error": str(err), "commands": []}

    def push_state(self, state: Dict[str, Any], source: str) -> Dict[str, Any]:
        payload = {
            "agent_id": self.agent_id,
            "source": source,
            "state": state,
        }
        return self._post(self.push_path, payload)

    def send_heartbeat(self, state: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "agent_id": self.agent_id,
            "status": state.get("validation_status"),
            "iteration_count": state.get("iteration_count", 0),
            "gpu_in_use": state.get("hardware_locks", {}).get("gpu_in_use", False),
        }
        return self._post(self.heartbeat_path, payload)

    def fetch_commands(self) -> List[Dict[str, Any]]:
        response = self._get(self.pull_path)
        commands = response.get("commands", [])
        return commands if isinstance(commands, list) else []

    @staticmethod
    def apply_command(state_manager, command: Dict[str, Any]) -> bool:
        """Apply common OpenClaw command payloads into SciOracle state.

        Supported command types:
        - set_conjecture: expects conjecture and optional generated_code
        - set_status: expects validation_status
        - state_patch: expects patch dictionary merged into state root
        """
        cmd_type = command.get("type")
        if cmd_type == "set_conjecture":
            state_manager.update_state({
                "current_conjecture": command.get("conjecture"),
                "generated_code": command.get("generated_code"),
                "validation_status": "validating",
            })
            return True

        if cmd_type == "set_status":
            state_manager.update_state({"validation_status": command.get("validation_status", "pending")})
            return True

        if cmd_type == "state_patch" and isinstance(command.get("patch"), dict):
            state_manager.update_state(command["patch"])
            return True

        return False
