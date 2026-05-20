from __future__ import annotations

import base64
import math
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from common.constants import DEFAULT_CLASSES
from server.services.control_logic import build_command, choose_target
from server.services.dataset_manager import DatasetManager
from server.services.model_manager import VisionEngine
from server.services.state import SystemState
from server.services.training_manager import TrainingManager
from server.utils.config import FRAMES_DIR, STORAGE_DIR

app = Flask(__name__)
state = SystemState()
vision = VisionEngine()
dataset_manager = DatasetManager()
training_manager = TrainingManager(state, dataset_manager)


def decode_image_from_request() -> np.ndarray:
    if "frame" in request.files:
        file = request.files["frame"]
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Не вдалося декодувати зображення з multipart")
        return frame

    data = request.json or {}
    frame_b64 = data.get("frame")
    if not frame_b64:
        raise ValueError("Кадр не передано")

    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]

    binary = base64.b64decode(frame_b64)
    arr = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Не вдалося декодувати base64 кадр")
    return frame


def save_frame(frame: np.ndarray) -> str:
    file_name = f"frame_{int(time.time() * 1000)}.jpg"
    path = FRAMES_DIR / file_name
    cv2.imwrite(str(path), frame)
    return file_name


def find_detection_for_click(detections: list[dict], x: int, y: int) -> dict | None:
    inside = []
    for det in detections:
        box = det.get("box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside.append(det)

    if inside:
        inside.sort(key=lambda item: item.get("area", 0), reverse=True)
        return inside[0]

    nearest = None
    nearest_distance = None
    for det in detections:
        center = det.get("center") or []
        if len(center) != 2:
            continue
        distance = math.dist((x, y), (center[0], center[1]))
        if nearest is None or distance < nearest_distance:
            nearest = det
            nearest_distance = distance

    if nearest is not None and nearest_distance is not None and nearest_distance <= 160:
        return nearest
    return None


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


@app.route("/api/vision/frame", methods=["POST"])
def api_vision_frame():
    try:
        frame = decode_image_from_request()
        detections = vision.detect(frame)
        manual_target = state.get_manual_target()
        target = choose_target(detections, manual_target)
        command_data = build_command(target, frame.shape[1], frame.shape[0])

        rendered = vision.annotate(frame, detections, target)
        rendered_name = save_frame(rendered)

        target_dict = target.to_dict() if target else None
        state.update_manual_target_tracking(target_dict)
        manual_target = state.get_manual_target()

        response = {
            "success": True,
            "detections": [det.to_dict() for det in detections],
            "target": target_dict,
            "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "control": command_data,
            "rendered_frame_url": f"/storage/frames/{rendered_name}",
            "model": vision.model_name,
            "server_time": time.time(),
            "manual_target": manual_target,
        }
        state.update_result(rendered_name, response)
        return jsonify(response)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/system/status")
def api_system_status():
    snapshot = state.get_snapshot()
    snapshot["dataset_stats"] = dataset_manager.get_dataset_stats()
    snapshot["uploads"] = dataset_manager.list_uploads()[:20]
    return jsonify(snapshot)


@app.route("/api/client/status", methods=["POST"])
def api_client_status():
    data = request.get_json(force=True, silent=True) or {}
    state.update_client_status(data)
    return jsonify({"success": True})


@app.route("/api/model/load", methods=["POST"])
def api_model_load():
    data = request.get_json(force=True) or {}
    model_name = data.get("model", "yolov8n.pt")
    vision.set_model(model_name)
    return jsonify({"success": True, "model": model_name})


@app.route("/api/target/select", methods=["POST"])
def api_target_select():
    data = request.get_json(force=True) or {}
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))

    latest_result = state.get_latest_result()
    detections = latest_result.get("detections") or []
    selected = find_detection_for_click(detections, x, y)
    if selected is None:
        return jsonify({"success": False, "error": "Об'єкт за вказаними координатами не знайдено"}), 404

    state.set_manual_target(selected)
    return jsonify({"success": True, "target": selected, "manual_target": state.get_manual_target()})


@app.route("/api/target/clear", methods=["POST"])
def api_target_clear():
    state.set_manual_target(None)
    return jsonify({"success": True, "manual_target": state.get_manual_target()})


@app.route("/api/annotator/upload", methods=["POST"])
def api_annotator_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Файл не передано"}), 400
    file = request.files["file"]
    file.filename = secure_filename(file.filename or "image.jpg")
    saved_name = dataset_manager.save_uploaded_image(file)
    return jsonify({
        "success": True,
        "file_name": saved_name,
        "file_url": f"/storage/uploads/{saved_name}",
    })




@app.route("/api/annotator/annotation")
def api_annotator_annotation():
    image_name = request.args.get("image_name", "").strip()
    if not image_name:
        return jsonify({"success": False, "error": "Не вказано image_name"}), 400
    annotation = dataset_manager.get_annotation(image_name)
    return jsonify({"success": True, "annotation": annotation})

@app.route("/api/annotator/delete", methods=["POST"])
def api_annotator_delete():
    data = request.get_json(force=True) or {}
    image_name = data.get("image_name", "")
    if not image_name:
        return jsonify({"success": False, "error": "Не вказано файл для видалення"}), 400
    result = dataset_manager.delete_image(image_name)
    return jsonify({"success": True, "deleted": result, "stats": dataset_manager.get_dataset_stats()})


@app.route("/api/annotator/save", methods=["POST"])
def api_annotator_save():
    data = request.get_json(force=True) or {}
    image_name = data.get("image_name")
    image_width = int(data.get("image_width", 0))
    image_height = int(data.get("image_height", 0))
    classes = data.get("classes") or []
    boxes = data.get("boxes") or []

    if not image_name or not image_width or not image_height:
        return jsonify({"success": False, "error": "Неповні дані зображення"}), 400

    record = dataset_manager.save_annotation(image_name, image_width, image_height, classes, boxes)
    return jsonify({"success": True, "record": record, "stats": dataset_manager.get_dataset_stats()})


@app.route("/api/training/start", methods=["POST"])
def api_training_start():
    data = request.get_json(force=True) or {}
    model_name = data.get("model", "yolov8n.pt")
    epochs = int(data.get("epochs", 10))
    imgsz = int(data.get("imgsz", 640))
    classes = [item.strip() for item in (data.get("classes") or []) if item.strip()]

    if not classes:
        return jsonify({"success": False, "error": "Потрібно вказати хоча б один клас"}), 400

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
    app.run(debug=True, host="0.0.0.0", port=5000)
