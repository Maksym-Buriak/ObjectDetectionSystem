from __future__ import annotations

import argparse
import time

import cv2
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot video client")
    parser.add_argument("--server", default="http://127.0.0.1:5000", help="Base URL сервера")
    parser.add_argument("--camera", default="0", help="Індекс камери або шлях до відеофайлу")
    parser.add_argument("--interval", type=float, default=0.2, help="Інтервал між відправками кадрів")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--show", action="store_true", help="Показувати локальне вікно")
    args = parser.parse_args()

    camera_source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Не вдалося відкрити джерело відео")

    session = requests.Session()
    last_send = 0.0

    print(f"[CLIENT] Підключено до {args.server}")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[CLIENT] Не вдалося отримати кадр")
            break

        now = time.time()
        result = None

        if now - last_send >= args.interval:
            ok_encode, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_encode:
                files = {"frame": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
                try:
                    response = session.post(f"{args.server}/api/vision/frame", files=files, timeout=30)
                    response.raise_for_status()
                    result = response.json()

                    client_status = {
                        "connected": True,
                        "last_seen": now,
                        "camera": str(camera_source),
                        "last_command": result.get("control", {}).get("command"),
                        "target": result.get("target"),
                    }
                    session.post(f"{args.server}/api/client/status", json=client_status, timeout=5)
                except Exception as exc:
                    print(f"[CLIENT] Помилка запиту: {exc}")
            last_send = now

        if result:
            command = result.get("control", {}).get("command")
            reason = result.get("control", {}).get("reason")
            target = result.get("target")
            print(f"[CLIENT] command={command} | reason={reason} | target={target.get('label') if target else '-'}")

        if args.show:
            display = frame.copy()
            cv2.putText(display, "Robot camera", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("Robot Client", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
