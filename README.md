# Quantix YOLO Object Detection

Fine-tune a small YOLO detector on a reproducible train/validation/test split of Ultralytics COCO128.

## Dataset

COCO128 is an 7 MB, 128-image COCO subset for smoke-testing object-detection workflows. The project downloads it into `data/raw/`, then makes a deterministic image-level 70/15/15 split under `data/splits/`. Dataset files and model weights are excluded from Git. See the [Ultralytics COCO128 dataset documentation](https://docs.ultralytics.com/datasets/detect/coco128) and cite the original [COCO paper](https://arxiv.org/abs/1405.0312).

The subset is too small for meaningful benchmark claims; use it to validate the pipeline before moving to full COCO.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --epochs 1 --device cpu
```

Outputs (including the best checkpoint) are written to `outputs/`.
