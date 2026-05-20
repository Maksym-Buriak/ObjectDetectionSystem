from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Iterable

from server.utils.config import (
    DATASET_DIR,
    DATASET_IMAGES_TRAIN,
    DATASET_IMAGES_VAL,
    DATASET_LABELS_TRAIN,
    DATASET_LABELS_VAL,
    UPLOADS_DIR,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetManager:
    def __init__(self) -> None:
        self.meta_path = DATASET_DIR / "annotations_index.json"
        if not self.meta_path.exists():
            self.meta_path.write_text("[]", encoding="utf-8")

    def save_uploaded_image(self, file_storage) -> str:
        suffix = Path(file_storage.filename or "image.jpg").suffix or ".jpg"
        file_name = f"{uuid.uuid4().hex}{suffix}"
        destination = UPLOADS_DIR / file_name
        file_storage.save(destination)
        return file_name

    def list_uploads(self) -> list[dict]:
        files = []
        for path in sorted(UPLOADS_DIR.iterdir(), reverse=True):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                files.append({
                    "name": path.name,
                    "url": f"/storage/uploads/{path.name}",
                })
        return files

    def get_annotation(self, image_name: str) -> dict:
        record = next((item for item in reversed(self._read_meta()) if item.get("image_name") == image_name), None)
        if not record:
            return {
                "image_name": image_name,
                "classes": [],
                "boxes": [],
                "image_width": None,
                "image_height": None,
                "split": None,
            }

        return {
            "image_name": image_name,
            "classes": record.get("classes") or [],
            "boxes": record.get("boxes") or [],
            "image_width": record.get("image_width"),
            "image_height": record.get("image_height"),
            "split": record.get("split"),
        }

    def save_annotation(self, image_name: str, image_width: int, image_height: int, classes: list[str], boxes: list[dict]) -> dict:
        classes = [name.strip() for name in classes if str(name).strip()]
        class_map = {name: idx for idx, name in enumerate(classes)}
        src_image = UPLOADS_DIR / image_name
        if not src_image.exists():
            raise FileNotFoundError(f"Не знайдено {image_name}")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Некоректні розміри зображення")

        normalized_boxes = []
        for box in boxes:
            cls_name = str(box.get("label", "")).strip()
            if cls_name not in class_map:
                continue

            x1 = max(0, min(int(round(float(box.get("x1", 0)))), image_width - 1))
            y1 = max(0, min(int(round(float(box.get("y1", 0)))), image_height - 1))
            x2 = max(0, min(int(round(float(box.get("x2", 0)))), image_width - 1))
            y2 = max(0, min(int(round(float(box.get("y2", 0)))), image_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            bw = x2 - x1
            bh = y2 - y1
            cx = x1 + bw / 2
            cy = y1 + bh / 2

            cx_n = cx / image_width
            cy_n = cy / image_height
            bw_n = bw / image_width
            bh_n = bh / image_height

            normalized_boxes.append({
                "label": cls_name,
                "class_id": class_map[cls_name],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "normalized": [cx_n, cy_n, bw_n, bh_n],
            })

        record = {
            "image_name": image_name,
            "saved_image": src_image.name,
            "label_file": f"{Path(image_name).stem}.txt",
            "split": "train",
            "classes": classes,
            "boxes_count": len(normalized_boxes),
            "boxes": normalized_boxes,
            "image_width": image_width,
            "image_height": image_height,
        }

        records = [item for item in self._read_meta() if item.get("image_name") != image_name]
        records.append(record)
        self._write_meta(records)
        self._rebuild_dataset_from_meta()

        refreshed = next((item for item in reversed(self._read_meta()) if item.get("image_name") == image_name), record)
        return refreshed

    def delete_image(self, image_name: str) -> dict:
        stem = Path(image_name).stem
        removed_files = []

        for path in [
            UPLOADS_DIR / image_name,
            DATASET_IMAGES_TRAIN / image_name,
            DATASET_IMAGES_VAL / image_name,
            DATASET_LABELS_TRAIN / f"{stem}.txt",
            DATASET_LABELS_VAL / f"{stem}.txt",
        ]:
            if path.exists() and path.is_file():
                path.unlink()
                removed_files.append(path.name)

        records = [item for item in self._read_meta() if item.get("image_name") != image_name]
        self._write_meta(records)
        self._rebuild_dataset_from_meta()

        return {
            "image_name": image_name,
            "removed_files": removed_files,
        }

    def get_dataset_stats(self) -> dict:
        return {
            "train_images": len([p for p in DATASET_IMAGES_TRAIN.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]),
            "val_images": len([p for p in DATASET_IMAGES_VAL.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]),
            "train_labels": len(list(DATASET_LABELS_TRAIN.glob("*.txt"))),
            "val_labels": len(list(DATASET_LABELS_VAL.glob("*.txt"))),
        }

    def create_dataset_yaml(self, classes: Iterable[str]) -> Path:
        classes = [cls.strip() for cls in classes if cls.strip()]
        yaml_path = DATASET_DIR / "dataset.yaml"
        content = [
            f"path: {DATASET_DIR.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(classes)}",
            "names:",
        ]
        content.extend([f"  {idx}: {name}" for idx, name in enumerate(classes)])
        yaml_path.write_text("\n".join(content), encoding="utf-8")
        return yaml_path

    def prepare_training_dataset(self) -> dict:
        self._rebuild_dataset_from_meta()
        stats = self.get_dataset_stats()
        actions = []

        if stats["train_images"] == 0 and stats["val_images"] == 0:
            raise RuntimeError("У датасеті немає жодного розміченого зображення. Спочатку завантаж і збережи розмітку хоча б для 1 файла.")
        if stats["train_labels"] == 0:
            raise RuntimeError("У train немає жодного label-файла. Спочатку збережи розмітку.")
        if stats["val_labels"] == 0:
            raise RuntimeError("У val немає жодного label-файла. Додай ще хоча б одне розмічене зображення або перезбережи розмітку.")

        if stats["val_images"] == 0 and stats["train_images"] > 0:
            sample = sorted([p for p in DATASET_IMAGES_TRAIN.iterdir() if p.is_file()])[0]
            self._copy_pair(sample, from_split="train", to_split="val")
            actions.append(f"Створено val-копію з {sample.name}, оскільки val був порожній.")
            stats = self.get_dataset_stats()

        return {"stats": stats, "actions": actions}

    def _copy_pair(self, image_path: Path, from_split: str, to_split: str) -> None:
        stem = image_path.stem
        src_label = (DATASET_LABELS_TRAIN if from_split == "train" else DATASET_LABELS_VAL) / f"{stem}.txt"
        dst_image = (DATASET_IMAGES_TRAIN if to_split == "train" else DATASET_IMAGES_VAL) / image_path.name
        dst_label = (DATASET_LABELS_TRAIN if to_split == "train" else DATASET_LABELS_VAL) / f"{stem}.txt"
        if not dst_image.exists():
            shutil.copy2(image_path, dst_image)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        elif not dst_label.exists():
            dst_label.write_text("", encoding="utf-8")

    def _write_meta(self, records: list) -> None:
        self.meta_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def _clear_dataset_dirs(self) -> None:
        for directory in [DATASET_IMAGES_TRAIN, DATASET_IMAGES_VAL, DATASET_LABELS_TRAIN, DATASET_LABELS_VAL]:
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()

    def _rebuild_dataset_from_meta(self) -> None:
        records = [item for item in self._read_meta() if (item.get("image_name") and (item.get("boxes") or []))]
        self._clear_dataset_dirs()
        if not records:
            return

        records = sorted(records, key=lambda item: item.get("image_name", ""))
        total = len(records)
        if total == 1:
            train_records = records
            val_records = records
        else:
            val_count = max(1, round(total * 0.2))
            val_positions = {
                min(total - 1, max(0, round((i + 1) * total / (val_count + 1)) - 1))
                for i in range(val_count)
            }
            val_records = [record for idx, record in enumerate(records) if idx in val_positions]
            train_records = [record for idx, record in enumerate(records) if idx not in val_positions]
            if not train_records:
                train_records = records[:-1]
                val_records = [records[-1]]

        for record in train_records:
            record["split"] = "train"
            self._materialize_record(record, split="train")
        for record in val_records:
            if record not in train_records:
                record["split"] = "val"
            else:
                record["split"] = "train+val"
            self._materialize_record(record, split="val")

        self._write_meta(records)

    def _materialize_record(self, record: dict, split: str) -> None:
        src_image = UPLOADS_DIR / record["image_name"]
        if not src_image.exists():
            return
        dst_image_dir = DATASET_IMAGES_TRAIN if split == "train" else DATASET_IMAGES_VAL
        dst_label_dir = DATASET_LABELS_TRAIN if split == "train" else DATASET_LABELS_VAL
        shutil.copy2(src_image, dst_image_dir / src_image.name)

        yolo_lines = []
        for box in record.get("boxes") or []:
            cls_id = int(box.get("class_id", 0))
            normalized = box.get("normalized") or [0, 0, 0, 0]
            if len(normalized) != 4:
                continue
            yolo_lines.append(
                f"{cls_id} {float(normalized[0]):.6f} {float(normalized[1]):.6f} {float(normalized[2]):.6f} {float(normalized[3]):.6f}"
            )
        (dst_label_dir / record["label_file"]).write_text("\n".join(yolo_lines), encoding="utf-8")

    def _read_meta(self) -> list:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return []
