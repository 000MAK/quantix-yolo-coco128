#!/usr/bin/env python3
"""Download COCO128, split it, and train a compact YOLO detector."""

from __future__ import annotations

import argparse
import json
import shutil
import os
import urllib.request
import zipfile
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / "data" / "ultralytics-config"))
from ultralytics import YOLO, settings

settings.update(
    {
        "datasets_dir": str(PROJECT_DIR / "data" / "ultralytics-datasets"),
        "weights_dir": str(PROJECT_DIR / "data" / "weights"),
        "runs_dir": str(PROJECT_DIR / "data" / "runs"),
        "sync": False,
    }
)


DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device such as 0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_and_prepare(data_dir: Path, seed: int) -> tuple[Path, dict[str, int]]:
    raw_dir = data_dir / "raw"
    split_dir = data_dir / "splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "coco128.zip"
    if not archive_path.exists():
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Quantix-Research/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    source_dir = raw_dir / "coco128"
    if not source_dir.exists():
        with zipfile.ZipFile(archive_path) as archive:
            root = Path(archive.namelist()[0].split("/")[0])
            for member in archive.infolist():
                target = Path(member.filename)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError(f"Unsafe path inside dataset archive: {member.filename}")
            archive.extractall(raw_dir)
        extracted = raw_dir / str(root)
        if extracted != source_dir:
            extracted.replace(source_dir)

    images = sorted((source_dir / "images" / "train2017").glob("*.jpg"))
    if len(images) < 10:
        raise ValueError(f"Expected 128 source images, found {len(images)}")
    data_yaml = split_dir / "dataset.yaml"
    if data_yaml.exists():
        counts = {
            split: len(list((split_dir / "images" / split).glob("*.jpg")))
            for split in ("train", "val", "test")
        }
        if all(counts.values()):
            return data_yaml, counts
        raise RuntimeError(f"Found an incomplete existing split at {split_dir}; review it before rebuilding.")
    indexes = list(range(len(images)))
    train, remainder = train_test_split(indexes, test_size=0.30, random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.50, random_state=seed)
    split_indexes = {"train": train, "val": validation, "test": test}

    counts: dict[str, int] = {}
    for split, split_ids in split_indexes.items():
        counts[split] = len(split_ids)
        target_images = split_dir / "images" / split
        target_labels = split_dir / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        for index in split_ids:
            source_image = images[index]
            label_path = source_dir / "labels" / "train2017" / f"{source_image.stem}.txt"
            shutil.copy2(source_image, target_images / source_image.name)
            if label_path.exists():
                shutil.copy2(label_path, target_labels / label_path.name)

    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(split_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml, counts


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    data_yaml, counts = download_and_prepare(args.data_dir, args.seed)
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=0,
        project=str(args.output_dir.resolve()),
        name="coco128-yolo11n",
        exist_ok=False,
        seed=args.seed,
        plots=True,
    )
    evaluation = model.val(data=str(data_yaml), split="test", device=args.device, workers=0)
    print(f"Image split counts: {counts}")
    print("Held-out test metrics:", evaluation.results_dict)
    test_metrics = {key: float(value) for key, value in evaluation.results_dict.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps({"split": "test", "image_counts": counts, "metrics": test_metrics}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
