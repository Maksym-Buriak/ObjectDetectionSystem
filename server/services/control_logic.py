from __future__ import annotations

from typing import Optional

from .model_manager import Detection

PRIORITY_CLASSES = ["person", "car", "bicycle"]


def _priority_value(label: str) -> int:
    return PRIORITY_CLASSES.index(label) if label in PRIORITY_CLASSES else 999


def choose_target(detections: list[Detection], manual_lock: Optional[dict] = None) -> Optional[Detection]:
    if not detections:
        return None

    if manual_lock and manual_lock.get("active"):
        target = _choose_locked_target(detections, manual_lock)
        if target is not None:
            return target

    def sort_key(det: Detection) -> tuple:
        return (_priority_value(det.label), -det.area, -det.confidence)

    return sorted(detections, key=sort_key)[0]


def _choose_locked_target(detections: list[Detection], manual_lock: dict) -> Optional[Detection]:
    preferred_label = manual_lock.get("label")
    last_center = manual_lock.get("last_center") or [None, None]
    lock_x, lock_y = last_center[0], last_center[1]

    def score(det: Detection) -> tuple:
        label_penalty = 0 if not preferred_label or det.label == preferred_label else 1
        if lock_x is None or lock_y is None:
            distance = 0
        else:
            distance = (det.cx - lock_x) ** 2 + (det.cy - lock_y) ** 2
        return (label_penalty, distance, -det.area, -det.confidence)

    return sorted(detections, key=score)[0]


def build_command(target: Optional[Detection], frame_width: int, frame_height: int) -> dict:
    center_x = frame_width // 2
    center_y = frame_height // 2

    if target is None:
        return {
            "command": "SEARCH",
            "reason": "Ціль не знайдена",
            "target_center": None,
            "offset": None,
        }

    offset_x = target.cx - center_x
    offset_y = target.cy - center_y
    normalized_x = offset_x / max(frame_width, 1)
    normalized_y = offset_y / max(frame_height, 1)

    horizontal_threshold = frame_width * 0.08

    if abs(offset_x) > horizontal_threshold:
        command = "TURN_RIGHT" if offset_x > 0 else "TURN_LEFT"
        reason = "Ціль зміщена відносно центру кадру"
    elif target.area < (frame_width * frame_height) * 0.06:
        command = "FORWARD"
        reason = "Ціль у центрі, але ще далека"
    else:
        command = "HOLD"
        reason = "Ціль зафіксована"

    return {
        "command": command,
        "reason": reason,
        "target_center": [target.cx, target.cy],
        "offset": [offset_x, offset_y],
        "normalized_offset": [round(normalized_x, 4), round(normalized_y, 4)],
    }
