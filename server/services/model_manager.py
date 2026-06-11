from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    cls_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: Optional[int] = None

    @property
    def cx(self) -> int:
        return int((self.x1 + self.x2) / 2)

    @property
    def cy(self) -> int:
        return int((self.y1 + self.y2) / 2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        return {
            "cls_id": self.cls_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": [self.x1, self.y1, self.x2, self.y2],
            "center": [self.cx, self.cy],
            "area": self.area,
            "track_id": self.track_id,
        }


class VisionEngine:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self.tracker_config = "bytetrack.yaml"

    def ensure_model(self) -> YOLO:
        if self.model is None:
            self.model = YOLO(self.model_name)
        return self.model

    def set_model(self, model_path: str) -> None:
        self.model_name = model_path
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, conf: float = 0.35) -> List[Detection]:
        model = self.ensure_model()
        results = model.track(
            source=frame,
            verbose=False,
            conf=conf,
            persist=True,
            tracker=self.tracker_config,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: List[Detection] = []

        if result.boxes is None:
            return detections

        ids = None
        if getattr(result.boxes, "id", None) is not None:
            ids = result.boxes.id.int().cpu().tolist()

        for idx, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            track_id = ids[idx] if ids and idx < len(ids) else None
            detections.append(
                Detection(
                    cls_id=cls_id,
                    label=str(names.get(cls_id, cls_id)),
                    confidence=score,
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                    track_id=track_id,
                )
            )
        return detections

    def annotate(self, frame: np.ndarray, detections: List[Detection], target: Optional[Detection] = None) -> np.ndarray:
        rendered = frame.copy()
        for det in detections:
            color = (76, 201, 240)
            thickness = 2
            is_target = False
            if target:
                if det.track_id is not None and target.track_id is not None:
                    is_target = det.track_id == target.track_id
                else:
                    is_target = det.x1 == target.x1 and det.y1 == target.y1 and det.x2 == target.x2 and det.y2 == target.y2
            if is_target:
                color = (99, 102, 241)
                thickness = 3
            cv2.rectangle(rendered, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
            id_part = f" #{det.track_id}" if det.track_id is not None else ""
            text = f"{det.label}{id_part} {det.confidence:.2f}"
            cv2.putText(rendered, text, (det.x1, max(20, det.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.circle(rendered, (det.cx, det.cy), 4, color, -1)
        return rendered
