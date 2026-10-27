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

## Results

Audit run on a 89 / 19 / 20 image split of COCO128: test precision **0.60**, recall **0.54**, mAP50 **0.620**, mAP50-95 **0.484** (20 test images). COCO128 is drawn from the COCO training set that the pretrained YOLO weights were trained on, so these scores are optimistic and not a measure of generalisation.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
