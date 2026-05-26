from __future__ import annotations

import base64
import io
import math
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
from PIL import Image
from werkzeug.utils import secure_filename

from common.constants import DEFAULT_CLASSES
from server.services.control_logic import build_command, choose_target
from server.services.dataset_manager import DatasetManager
from server.services.model_manager import VisionEngine
from server.services.state import SystemState
from server.services.training_manager import TrainingManager
from server.utils.config import STORAGE_DIR

# --- Налаштування додатку та SocketIO ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key_for_socketio"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- Глобальні сервіси ---
state = SystemState()
vision = VisionEngine()
dataset_manager = DatasetManager()
training_manager = TrainingManager(state, dataset_manager)

# --- Новий мульти-пристроєвий FrameHandler ---
class MultiFrameHandler:
    def __init__(self):
        self.frames = {}          # Структура: { source_id: frame_ndarray }
        self.sources_info = {}    # Структура: { source_id: { "name": str, "last_seen": float, "sid": str } }
        self.active_source = None # ID джерела, яке ми зараз процесимо та виводимо на Dashboard
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def set_frame(self, source_id: str, source_name: str, sid: str, frame: np.ndarray):
        with self.lock:
            self.frames[source_id] = frame
            self.sources_info[source_id] = {
                "name": source_name,
                "last_seen": time.time(),
                "sid": sid
            }
            # Якщо активне джерело ще не обрано, автоматично ставимо перше підключене
            if self.active_source is None:
                self.active_source = source_id
        self.new_frame_event.set()

    def get_active_frame(self) -> tuple[np.ndarray | None, str | None, str | None]:
        """Повертає кадр, ID та зрозуміле ім'я поточного АКТИВНОГО пристрою."""
        with self.lock:
            if self.active_source is None or self.active_source not in self.frames:
                return None, None, None
            
            frame = self.frames.get(self.active_source)
            
            # Очищуємо кадр після вичитки, щоб не обробляти статичну картинку в циклі
            if frame is not None:
                self.frames[self.active_source] = None
                
            source_id = self.active_source
            source_name = self.sources_info[source_id]["name"]
            
        return frame, source_id, source_name

    def clean_dead_sources(self, timeout=5.0):
        """Видаляє пристрої, від яких давно не було кадрів."""
        with self.lock:
            now = time.time()
            dead_ids = [
                sid for sid, info in self.sources_info.items() 
                if now - info["last_seen"] > timeout
            ]
            for sid in dead_ids:
                self.frames.pop(sid, None)
                self.sources_info.pop(sid, None)
                if self.active_source == sid:
                    self.active_source = list(self.sources_info.keys())[0] if self.sources_info else None

    def get_available_sources(self) -> list[dict]:
        with self.lock:
            return [
                {"id": sid, "name": info["name"], "is_active": (sid == self.active_source)}
                for sid, info in self.sources_info.items()
            ]

    def set_active_source(self, source_id: str) -> bool:
        with self.lock:
            if source_id in self.sources_info:
                self.active_source = source_id
                return True
            return False

frame_handler = MultiFrameHandler()

# --- Фонова обробка ---

def process_frame_and_broadcast(frame: np.ndarray, source_id: str, source_name: str):
    state.update_client_status({"camera": source_name, "timestamp": time.time()})

    detections = vision.detect(frame)
    manual_target = state.get_manual_target()
    target = choose_target(detections, manual_target)
    command_data = build_command(target, frame.shape[1], frame.shape[0])

    rendered_frame = vision.annotate(frame, detections, target)

    _, buffer = cv2.imencode('.jpg', rendered_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    target_dict = target.to_dict() if target else None
    state.update_manual_target_tracking(target_dict)
    manual_target = state.get_manual_target()

    response = {
        "success": True,
        "detections": [det.to_dict() for det in detections],
        "target": target_dict,
        "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "control": command_data,
        "rendered_frame": f"data:image/jpeg;base64,{frame_b64}",
        "model": vision.model_name,
        "server_time": time.time(),
        "manual_target": manual_target,
        "source": source_name,
        "source_id": source_id
    }
    state.update_result(None, response)
    socketio.emit("dashboard_update", response)
    
    # ВІДПРАВКА КОМАНДИ НАЗАД КОНКРЕТНОМУ РОБОТУ/ПРИСТРОЮ
    with frame_handler.lock:
        active_info = frame_handler.sources_info.get(source_id)
        active_sid = active_info["sid"] if active_info else None
        
    if active_sid:
        socketio.emit("control_response", {"control": command_data}, to=active_sid)

def frame_processing_loop():
    counter = 0
    while True:
        frame_handler.new_frame_event.wait(timeout=1.0)
        frame, src_id, src_name = frame_handler.get_active_frame()
        frame_handler.new_frame_event.clear()

        if frame is not None and src_id is not None:
            try:
                process_frame_and_broadcast(frame, src_id, src_name)
            except Exception as e:
                print(f"Помилка у фоновому обробнику кадрів: {e}")
        
        # Раз на 100 ітерацій чистимо "відпалі" пристрої
        counter += 1
        if counter % 100 == 0:
            frame_handler.clean_dead_sources()
            socketio.emit("available_sources", frame_handler.get_available_sources())
            
        socketio.sleep(0.01)

# --- Обробники WebSocket ---

@socketio.on("connect")
def handle_connect():
    print(f"Клієнт підключився: {request.sid}")
    # Відразу відправляємо список пристроїв новому клієнту
    socketio.emit("available_sources", frame_handler.get_available_sources(), to=request.sid)

@socketio.on("disconnect")
def handle_disconnect():
    print(f"Клієнт відключився: {request.sid}")

@socketio.on("video_frame")
def handle_video_frame(data):
    try:
        # Спроба визначити унікальний ID пристрою
        if isinstance(data, bytes):
            source_id = f"android_{request.sid}"
            source_name = "Android Camera"
            frame_data = data
        else:
            device_id = data.get("device_id")
            source_id = device_id if device_id else f"webcam_{request.sid}"
            source_name = data.get("source", "Webcam")
            frame_data = data.get("frame")

        if isinstance(frame_data, bytes):
            pil_image = Image.open(io.BytesIO(frame_data))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif isinstance(frame_data, str):
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]
            binary = base64.b64decode(frame_data)
            arr = np.frombuffer(binary, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            return

        if frame is None:
            return

        # Зберігаємо кадр у нашому менеджері
        frame_handler.set_frame(source_id, source_name, request.sid, frame)

    except Exception as e:
        print(f"Помилка декодування WebSocket кадру: {e}")

# --- Нові API ендпоінти для фронтенду ---

@app.route("/api/sources", methods=["GET"])
def get_sources():
    """Повертає список усіх підключених зараз пристроїв-камер."""
    return jsonify({"success": True, "sources": frame_handler.get_available_sources()})

@app.route("/api/sources/select", methods=["POST"])
def select_source():
    """Ендпоінт для перемикання активної камери з фронтенду."""
    data = request.get_json(force=True) or {}
    source_id = data.get("source_id")
    if not source_id:
        return jsonify({"success": False, "error": "Не вказано source_id"}), 400
        
    if frame_handler.set_active_source(source_id):
        # Оповіщаємо всі інтерфейси про зміну активного стримера
        socketio.emit("available_sources", frame_handler.get_available_sources())
        return jsonify({"success": True, "active_source": source_id})
    return jsonify({"success": False, "error": "Пристрій не знайдено або відключено"}), 404

# --- Інші HTTP ендпоінти ---

def find_detection_for_click(detections: list[dict], x: int, y: int) -> dict | None:
    inside = []
    for det in detections:
        box = det.get("box") or []
        if len(box) != 4: continue
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside.append(det)
    if inside:
        inside.sort(key=lambda item: item.get("area", 0), reverse=True)
        return inside[0]
    nearest, nearest_distance = None, None
    for det in detections:
        center = det.get("center") or []
        if len(center) != 2: continue
        distance = math.dist((x, y), (center[0], center[1]))
        if nearest is None or distance < nearest_distance:
            nearest, nearest_distance = det, distance
    return nearest if nearest is not None and nearest_distance <= 160 else None

@app.route("/")
def index():
    return render_template("index.html", classes=DEFAULT_CLASSES)

@app.route("/robot-client")
def robot_client():
    return render_template("robot_client.html")

@app.route("/annotator")
def annotator():
    uploads = dataset_manager.list_uploads()
    stats = dataset_manager.get_dataset_stats()
    return render_template("annotator.html", uploads=uploads, stats=stats)

@app.route("/training")
def training():
    return render_template("training.html", stats=dataset_manager.get_dataset_stats(), default_classes=",".join(DEFAULT_CLASSES))

@app.route("/storage/<path:subpath>")
def storage_files(subpath: str):
    return send_from_directory(STORAGE_DIR, subpath)

@app.route("/api/system/status")
def api_system_status():
    snapshot = state.get_snapshot()
    snapshot["dataset_stats"] = dataset_manager.get_dataset_stats()
    snapshot["uploads"] = dataset_manager.list_uploads()[:20]
    # Додаємо список пристроїв до загального статусу
    snapshot["active_sources"] = frame_handler.get_available_sources()
    return jsonify(snapshot)

@app.route("/api/model/load", methods=["POST"])
def api_model_load():
    data = request.get_json(force=True) or {}
    model_name = data.get("model", "yolov8n.pt")
    vision.set_model(model_name)
    return jsonify({"success": True, "model": model_name})

@app.route("/api/target/select", methods=["POST"])
def api_target_select():
    data = request.get_json(force=True) or {}
    x, y = int(data.get("x", -1)), int(data.get("y", -1))
    latest_result = state.get_latest_result()
    detections = latest_result.get("detections") or []
    selected = find_detection_for_click(detections, x, y)
    if selected is None:
        return jsonify({"success": False, "error": "Об'єкт за вказаними координатами не знайдено"}), 404
    state.set_manual_target(selected)
    socketio.emit("dashboard_update", {"manual_target": state.get_manual_target(), "source": state.get_snapshot().get("latest_client_status", {}).get("camera")})
    return jsonify({"success": True, "target": selected, "manual_target": state.get_manual_target()})

@app.route("/api/target/clear", methods=["POST"])
def api_target_clear():
    state.set_manual_target(None)
    socketio.emit("dashboard_update", {"manual_target": state.get_manual_target(), "source": state.get_snapshot().get("latest_client_status", {}).get("camera")})
    return jsonify({"success": True, "manual_target": state.get_manual_target()})

@app.route("/api/annotator/upload", methods=["POST"])
def api_annotator_upload():
    if "file" not in request.files: return jsonify({"success": False, "error": "Файл не передано"}), 400
    file = request.files["file"]
    file.filename = secure_filename(file.filename or "image.jpg")
    saved_name = dataset_manager.save_uploaded_image(file)
    return jsonify({"success": True, "file_name": saved_name, "file_url": f"/storage/uploads/{saved_name}"})

@app.route("/api/annotator/annotation")
def api_annotator_annotation():
    image_name = request.args.get("image_name", "").strip()
    if not image_name: return jsonify({"success": False, "error": "Не вказано image_name"}), 400
    annotation = dataset_manager.get_annotation(image_name)
    return jsonify({"success": True, "annotation": annotation})

@app.route("/api/annotator/delete", methods=["POST"])
def api_annotator_delete():
    data = request.get_json(force=True) or {}
    image_name = data.get("image_name", "")
    if not image_name: return jsonify({"success": False, "error": "Не вказано файл для видалення"}), 400
    result = dataset_manager.delete_image(image_name)
    return jsonify({"success": True, "deleted": result, "stats": dataset_manager.get_dataset_stats()})

@app.route("/api/annotator/save", methods=["POST"])
def api_annotator_save():
    data = request.get_json(force=True) or {}
    image_name, image_width, image_height = data.get("image_name"), int(data.get("image_width", 0)), int(data.get("image_height", 0))
    classes, boxes = data.get("classes") or [], data.get("boxes") or []
    if not image_name or not image_width or not image_height: return jsonify({"success": False, "error": "Неповні дані зображення"}), 400
    record = dataset_manager.save_annotation(image_name, image_width, image_height, classes, boxes)
    return jsonify({"success": True, "record": record, "stats": dataset_manager.get_dataset_stats()})

@app.route("/api/training/start", methods=["POST"])
def api_training_start():
    data = request.get_json(force=True) or {}
    model_name, epochs, imgsz = data.get("model", "yolov8n.pt"), int(data.get("epochs", 10)), int(data.get("imgsz", 640))
    classes = [item.strip() for item in (data.get("classes") or []) if item.strip()]
    if not classes: return jsonify({"success": False, "error": "Потрібно вказати хоча б один клас"}), 400
    try:
        training_manager.start_training(model_name=model_name, classes=classes, epochs=epochs, imgsz=imgsz)
        return jsonify({"success": True, "message": "Тренування запущено"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/training/status")
def api_training_status():
    status = dict(state.get_snapshot().get("training_status", {}))
    status["dataset_stats"] = dataset_manager.get_dataset_stats()
    return jsonify(status)

if __name__ == "__main__":
    socketio.start_background_task(target=frame_processing_loop)
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
