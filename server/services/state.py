from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class SystemState:
    latest_frame_path: str = ""
    latest_frame_at: float = 0.0
    latest_result: dict = field(default_factory=dict)
    latest_client_status: dict = field(default_factory=dict)
    training_status: dict = field(default_factory=lambda: {
        "running": False,
        "message": "Тренування ще не запускалось",
        "last_started_at": None,
        "last_finished_at": None,
        "run_dir": None,
        "logs": [],
        "history": [],
        "stdout_tail": "",
        "stderr_tail": "",
    })
    manual_target: dict = field(default_factory=lambda: {
        "active": False,
        "label": None,
        "last_center": None,
        "last_box": None,
        "selected_at": None,
        "lost": False,
    })
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_result(self, frame_path: str, result: dict) -> None:
        with self.lock:
            self.latest_frame_path = frame_path
            self.latest_frame_at = time.time()
            self.latest_result = result

    def update_client_status(self, status: dict) -> None:
        with self.lock:
            self.latest_client_status = status

    def set_manual_target(self, target: Optional[dict]) -> None:
        with self.lock:
            if not target:
                self.manual_target = {
                    "active": False,
                    "label": None,
                    "last_center": None,
                    "last_box": None,
                    "selected_at": None,
                    "lost": False,
                }
                return

            self.manual_target = {
                "active": True,
                "label": target.get("label"),
                "last_center": target.get("center"),
                "last_box": target.get("box"),
                "selected_at": time.time(),
                "lost": False,
            }

    def update_manual_target_tracking(self, target: Optional[dict]) -> None:
        with self.lock:
            if not self.manual_target.get("active"):
                return
            if target is None:
                self.manual_target["lost"] = True
                return
            self.manual_target["label"] = target.get("label")
            self.manual_target["last_center"] = target.get("center")
            self.manual_target["last_box"] = target.get("box")
            self.manual_target["lost"] = False

    def get_manual_target(self) -> dict:
        with self.lock:
            return dict(self.manual_target)

    def get_latest_result(self) -> dict:
        with self.lock:
            return dict(self.latest_result)

    def get_snapshot(self) -> dict:
        with self.lock:
            return {
                "latest_frame_path": self.latest_frame_path,
                "latest_frame_at": self.latest_frame_at,
                "latest_result": self.latest_result,
                "latest_client_status": self.latest_client_status,
                "training_status": self.training_status,
                "manual_target": self.manual_target,
            }

    def set_training(self, **kwargs) -> None:
        with self.lock:
            self.training_status.update(kwargs)

    def append_training_log(self, message: str) -> None:
        with self.lock:
            logs = list(self.training_status.get("logs") or [])
            logs.append(f"[{_now_text()}] {message}")
            self.training_status["logs"] = logs[-400:]

    def finish_training_history(self, status: str, message: str, run_dir: Optional[str] = None) -> None:
        with self.lock:
            history = list(self.training_status.get("history") or [])
            history.append({
                "finished_at": _now_text(),
                "status": status,
                "message": message,
                "run_dir": run_dir,
            })
            self.training_status["history"] = history[-50:]
