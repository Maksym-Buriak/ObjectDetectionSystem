from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO


def _force_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    try:
        print(f"[worker] python={sys.executable}", flush=True)
        print(f"[worker] model={args.model}", flush=True)
        print(f"[worker] data={args.data}", flush=True)
        print(f"[worker] epochs={args.epochs}, imgsz={args.imgsz}", flush=True)
        print(f"[worker] project={args.project}", flush=True)

        data_path = Path(args.data)
        print(f"[worker] dataset.yaml exists={data_path.exists()}", flush=True)
        if not data_path.exists():
            raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

        model = YOLO(args.model)
        print("[worker] YOLO model loaded", flush=True)

        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            project=args.project,
            exist_ok=True,
            verbose=True,
        )

        save_dir = getattr(results, "save_dir", None)
        if save_dir is None and isinstance(results, dict):
            save_dir = results.get("save_dir")

        print(f"RUN_DIR={save_dir or ''}", flush=True)

    except Exception as exc:
        print(f"[worker][ERROR] {type(exc).__name__}: {exc}", flush=True)
        print("[worker][TRACEBACK_BEGIN]", flush=True)
        print(traceback.format_exc(), flush=True)
        print("[worker][TRACEBACK_END]", flush=True)
        raise


if __name__ == "__main__":
    main()