from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
STORAGE_DIR = BASE_DIR / "storage"
FRAMES_DIR = STORAGE_DIR / "frames"
UPLOADS_DIR = STORAGE_DIR / "uploads"
DATASET_DIR = STORAGE_DIR / "datasets"
DATASET_IMAGES_TRAIN = DATASET_DIR / "images" / "train"
DATASET_IMAGES_VAL = DATASET_DIR / "images" / "val"
DATASET_LABELS_TRAIN = DATASET_DIR / "labels" / "train"
DATASET_LABELS_VAL = DATASET_DIR / "labels" / "val"
RUNS_DIR = STORAGE_DIR / "runs"

for path in [
    STORAGE_DIR,
    FRAMES_DIR,
    UPLOADS_DIR,
    DATASET_DIR,
    DATASET_IMAGES_TRAIN,
    DATASET_IMAGES_VAL,
    DATASET_LABELS_TRAIN,
    DATASET_LABELS_VAL,
    RUNS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
