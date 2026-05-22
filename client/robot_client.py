from __future__ import annotations

import argparse
import base64
import sys
import threading
import time
from pathlib import Path

import cv2
import socketio

# --- Налаштування шляхів ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Глобальні змінні ---
sio = socketio.Client()
last_control_command = {}
is_running = True

# --- Обробники подій Socket.IO ---
@sio.on("connect")
def on_connect():
    print("Підключено до сервера WebSocket.")

@sio.on("disconnect")
def on_disconnect():
    """Ця функція викликається, коли зв'язок з сервером втрачено."""
    global is_running
    if is_running:
        print("Відключено від сервера WebSocket. Завершення роботи клієнта...")
        is_running = False

@sio.on("dashboard_update")
def on_dashboard_update(data):
    global last_control_command
    if "control" in data:
        last_control_command = data["control"]

# --- Основна логіка клієнта ---
def send_frames(camera_index: int, show_window: bool, fps: int):
    global is_running
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити камеру {camera_index}")
        is_running = False
        return

    window_name = "Robot Camera"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_interval = 1.0 / fps

    while is_running:
        start_time = time.time()
        
        if not sio.connected:
            is_running = False
            break
            
        ret, frame = cap.read()
        if not ret:
            print("Помилка: не вдалося отримати кадр.")
            time.sleep(0.5)
            continue

        if sio.connected:
            try:
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                sio.emit("video_frame", {"frame": f"data:image/jpeg;base64,{frame_b64}", "source": "Webcam"})
            except Exception:
                is_running = False
                break

        if show_window:
            display_frame = frame.copy()
            cmd = last_control_command.get("command", "WAIT")
            reason = last_control_command.get("reason", "")
            cv2.putText(display_frame, f"CMD: {cmd}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if reason:
                cv2.putText(display_frame, reason, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            # Перевіряємо і натискання 'q', і стан is_running
            if cv2.waitKey(1) & 0xFF == ord('q') or not is_running:
                break
        
        elapsed_time = time.time() - start_time
        sleep_time = frame_interval - elapsed_time
        if sleep_time > 0:
            # Використовуємо sio.sleep для неблокуючої затримки
            sio.sleep(sleep_time)

    # --- Блок завершення роботи ---
    print("Звільнення ресурсів...")
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
        # Додатково викликаємо waitKey кілька разів, щоб OpenCV встиг обробити закриття вікна
        for _ in range(5):
            cv2.waitKey(1)
            
    if sio.connected:
        sio.disconnect()
    print("Камеру звільнено, вікна закрито.")

def main():
    global is_running
    parser = argparse.ArgumentParser(description="Клієнт для робота, що транслює відео на сервер.")
    parser.add_argument("--host", default="127.0.0.1", help="IP-адреса сервера.")
    parser.add_argument("--port", default=5000, type=int, help="Порт сервера.")
    parser.add_argument("--camera", default=0, type=int, help="Індекс камери для захоплення відео.")
    parser.add_argument("--show", action="store_true", help="Показувати вікно з відео.")
    parser.add_argument("--fps", default=15, type=int, help="Кількість кадрів в секунду для відправки.")
    args = parser.parse_args()

    server_url = f"http://{args.host}:{args.port}"
    
    try:
        sio.connect(server_url, wait_timeout=5)
    except socketio.exceptions.ConnectionError as e:
        print(f"Не вдалося підключитися до WebSocket: {e}")
        return

    if not sio.connected:
        print("Не вдалося встановити з'єднання.")
        return

    frame_thread = threading.Thread(target=send_frames, args=(args.camera, args.show, args.fps))
    frame_thread.start()
    
    try:
        while is_running and frame_thread.is_alive():
            frame_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        print("\nОтримано сигнал Ctrl+C. Завершення роботи...")
        is_running = False
    
    if frame_thread.is_alive():
        frame_thread.join()
    print("Програму повністю закрито.")

if __name__ == "__main__":
    main()
