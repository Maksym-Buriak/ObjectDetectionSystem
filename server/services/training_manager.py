from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from server.utils.config import PROJECT_DIR, RUNS_DIR
from .state import SystemState
from .dataset_manager import DatasetManager


class TrainingManager:
    def __init__(self, state: SystemState, dataset_manager: DatasetManager) -> None:
        self.state = state
        self.dataset_manager = dataset_manager
        self.thread: Optional[threading.Thread] = None

    def start_training(self, model_name: str, classes: list[str], epochs: int, imgsz: int) -> None:
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Тренування вже виконується")

        prep = self.dataset_manager.prepare_training_dataset()
        yaml_path = self.dataset_manager.create_dataset_yaml(classes)

        self.state.set_training(
            logs=[],
            stdout_tail="",
            stderr_tail="",
            run_dir=None,
            message="Підготовка до запуску завершена",
        )
        self.state.append_training_log(
            f"Початок підготовки навчання. model={model_name}, epochs={epochs}, imgsz={imgsz}"
        )

        for action in prep.get("actions", []):
            self.state.append_training_log(action)

        stats = prep.get("stats", {})
        self.state.append_training_log(
            "Dataset stats: "
            f"train_images={stats.get('train_images', 0)}, "
            f"val_images={stats.get('val_images', 0)}, "
            f"train_labels={stats.get('train_labels', 0)}, "
            f"val_labels={stats.get('val_labels', 0)}"
        )
        self.state.append_training_log(f"Згенеровано dataset.yaml: {yaml_path}")

        self.thread = threading.Thread(
            target=self._run_training,
            args=(model_name, yaml_path, epochs, imgsz),
            daemon=True,
        )
        self.thread.start()

    def _run_training(self, model_name: str, yaml_path: Path, epochs: int, imgsz: int) -> None:
        self.state.set_training(
            running=True,
            message="Тренування запущено",
            last_started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            last_finished_at=None,
            run_dir=None,
        )

        script = PROJECT_DIR / "server" / "services" / "training_worker.py"
        cmd = [
            sys.executable,
            str(script),
            "--model",
            model_name,
            "--data",
            str(yaml_path),
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--project",
            str(RUNS_DIR),
        ]
        self.state.append_training_log(f"Команда запуску: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        try:
            process = subprocess.run(
                cmd,
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                env=env,
            )

            stdout_text = process.stdout or ""
            stderr_text = process.stderr or ""

            stdout_tail = stdout_text[-20000:]
            stderr_tail = stderr_text[-20000:]

            self.state.set_training(stdout_tail=stdout_tail, stderr_tail=stderr_tail)
            self.state.append_training_log(f"Повернутий код процесу: {process.returncode}")

            if stdout_tail:
                self.state.append_training_log("stdout tail збережено у полі логів.")
                for line in stdout_tail.splitlines()[-40:]:
                    self.state.append_training_log(f"[stdout] {line}")

            if stderr_tail:
                self.state.append_training_log("stderr tail збережено у полі логів.")
                for line in stderr_tail.splitlines()[-40:]:
                    self.state.append_training_log(f"[stderr] {line}")

            if process.returncode != 0:
                raw_error = (stderr_tail or stdout_tail or "невідома помилка")[-4000:]
                message = f"Помилка тренування:\n{raw_error}"

                self.state.set_training(
                    running=False,
                    message="Тренування завершилось з помилкою",
                    last_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    run_dir=None,
                )
                self.state.append_training_log(message)
                self.state.finish_training_history(
                    "error",
                    "Тренування завершилось з помилкою",
                    run_dir=None,
                )
                return

            run_dir = None
            for line in stdout_text.splitlines():
                if line.startswith("RUN_DIR="):
                    run_dir = line.split("=", 1)[1].strip() or None

            message = "Тренування завершено успішно"
            self.state.set_training(
                running=False,
                message=message,
                last_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                run_dir=run_dir,
            )
            self.state.finish_training_history("success", message, run_dir=run_dir)

        except Exception as exc:
            message = f"Виняток під час тренування: {type(exc).__name__}: {exc}"
            self.state.append_training_log(message)
            self.state.set_training(
                running=False,
                message=message,
                last_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.state.finish_training_history("exception", message, run_dir=None)