# Phase 1 — Project Scaffold & Pose Extraction Pipeline

**Status:** Complete  
**Date:** 2026-09-06

---

## 1. What was implemented

### `backend/core/config.py`
Pydantic-settings `Settings` class. Single source of truth for all constants, paths, and hyperparameters. Module-level `settings` singleton imported everywhere. No magic numbers anywhere else in the codebase.

Key constants:
| Constant | Value |
|---|---|
| `num_keypoints` | 130 |
| `num_hand_landmarks` | 21 |
| `window_size` | 30 |
| `stride` | 15 |
| `confidence_threshold` | 0.85 |
| `pause_timeout_seconds` | 1.5 |
| `commit_streak` | 2 |

### `backend/core/logger.py`
Loguru-based logger factory with console (colourised) and file (rotating, 7-day retention) sinks. `get_logger(__name__)` used in every module.

### `backend/pose/extractor.py` — `KeypointExtractor`
MediaPipe Holistic wrapper. Produces exactly **130 float32 values** per frame:
- Left hand: 21 × 3 = 63
- Right hand: 21 × 3 = 63
- Pose context (left shoulder, right shoulder, left wrist, right wrist) x-coord only = 4

Zero-filled when hand/pose not detected. Supports context manager.

### `backend/pose/buffer.py` — `FrameBuffer`
Sliding window accumulator. Emits `(30, 130)` numpy arrays with 15-frame stride (50% overlap). Exposes `add()`, `reset()`, `current_length`, `is_full`.

### `backend/pose/visualizer.py` — `SkeletonVisualiser`
Debug rendering on BGR frames. ISL = teal (`#3ECFB2`), NGT = violet (`#A78BFA`). Draws hand connections, pose context dots, language badge, and confidence bar. Supports both live MediaPipe results and raw keypoint arrays.

### `backend/ml/data/loader.py` — `DatasetLoader`
Processes all four available datasets:

| Dataset | Method | Output files |
|---|---|---|
| INCLUDE | `process_include()` | `isl_sequences_include.npy` + `isl_labels_include.npy` |
| ISLVT | `process_islvt()` | `isl_sequences_islvt.npy` + `isl_labels_islvt.npy` |
| Indian Sign Language_Dataset | `process_isl_dataset()` | `isl_sequences_isl_dataset.npy` + `isl_labels_isl_dataset.npy` |
| NGT HoReCo 1.2 | `process_ngt_horeco()` | `ngt_sequences.npy` + `ngt_labels.npy` |

Extracts zips on the fly, processes every video frame-by-frame, segments into 30-frame windows, and prints per-class statistics.

### `scripts/collect_data.py`
CLI tool for recording self-signed data via webcam:
```
python scripts/collect_data.py --sign "hello" --samples 50 --language ISL
```
Shows countdown, live skeleton overlay, per-frame progress bar. Saves `data/self_recorded/{language}/{sign}/sequence_{N:04d}.npy`.

---

## 2. Dataset inventory

| Dataset | Language | Format | Videos | Notes |
|---|---|---|---|---|
| INCLUDE | ISL | 46 zip files | ~4292 | 263 sign classes, 15 categories. Zips not extracted yet. |
| ISLVT | ISL | .MOV flat + Excel gloss | 153 | ISL sentence videos. Excel (`marathi-English_ISLgloss1.xlsx`) for Phase 3. |
| Indian Sign Language_Dataset | ISL | ISL.zip | Unknown until extracted | Compressed — extracted by loader. |
| NGT HoReCo 1.2 | NGT | .mp4/.mov + Excel | 297 | Continuous signing. Excel annotation for Phase 3. |

> **Note:** Dataset statistics (exact sequence counts, samples per class) will be populated after the first `process_*()` run, since INCLUDE zips and ISL.zip are not yet extracted.

---

## 3. Keypoint extraction performance

To be measured during Phase 2 integration. Expected: ~30 fps on CPU with MediaPipe model_complexity=1.

Run benchmark:
```python
import time, cv2
from backend.pose.extractor import KeypointExtractor

cap = cv2.VideoCapture(0)
extractor = KeypointExtractor()
times = []
for _ in range(300):
    ret, frame = cap.read()
    t0 = time.perf_counter()
    extractor.extract(frame)
    times.append(time.perf_counter() - t0)
print(f"Mean FPS: {1 / (sum(times)/len(times)):.1f}")
```

---

## 4. Unit test results

Run with:
```powershell
(C:\Users\Deep\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate base)
cd signbridge
pytest backend/tests/ -v --tb=short
```

Tests written:
- `test_pose.py`: 12 tests — shape, dtype, zero-fill, validation, context manager, keypoint layout decomposition
- `test_buffer.py`: 14 tests — emission timing, stride overlap, state, reset, validation, custom dimensions

---

## 5. Known limitations

- **INCLUDE extraction time:** Extracting all 46 zips (~40 GB compressed) will take significant time and disk space. Run `process_include()` overnight.
- **ISLVT label granularity:** Labels are sentence IDs (numeric), not individual sign glosses. Phase 3 NLP will parse the Excel file for gloss-level annotations.
- **NGT HoReCo:** Videos contain continuous signing across multiple signs. Phase 2 will need temporal segmentation using the Excel annotation file.
- **ISL.zip structure unknown:** Internal directory layout will be confirmed on first extraction — the loader infers labels from parent folder names.
- **No augmentation yet:** Phase 2 will add random time-warping, mirroring, and noise injection.

---

## 6. What Phase 2 builds on this

- CNN + LSTM model (`backend/ml/models/`) using the `(N, 30, 130)` arrays produced by `DatasetLoader`
- `FrameBuffer` feeds directly into the model's inference loop
- `KeypointExtractor` stays unchanged — Phase 2 only adds the model layer on top
- Training configs in `backend/ml/training/configs/*.yaml`
- Evaluation reports in `backend/ml/evaluation/`
- Word accumulator (`backend/ml/inference/accumulator.py`) built on top of model output
