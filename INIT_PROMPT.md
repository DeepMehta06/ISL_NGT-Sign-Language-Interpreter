SignBridge — Claude Initialization Prompt

Copy and paste the entire block below to Claude (or any AI agent) to initialize the project structure for Phase 1.

---

## PROMPT START

You are initializing a production-grade machine learning project called **SignBridge** — a real-time, bidirectional sign language interpreter supporting ISL (Indian Sign Language) and NGT (Dutch Sign Language).

Before doing anything else, read and internalize these rules:

**Environment:** Every Python command must be run after activating conda:

```powershell
(C:\Users\Deep\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate base)
```

**Your task for Phase 1:** Set up the complete project scaffold and implement the pose extraction pipeline. Nothing beyond Phase 1 scope.

---

### Phase 1 scope — exactly this, nothing more

1. Create the complete folder structure as defined in ARCHITECTURE.md
2. Set up `backend/core/config.py` — all constants, paths, hyperparameters
3. Set up `backend/core/logger.py` — structured logging
4. Implement `backend/pose/extractor.py` — MediaPipe Holistic keypoint extraction
5. Implement `backend/pose/buffer.py` — sliding window frame buffer (30 frames, 15-frame stride)
6. Implement `backend/pose/visualizer.py` — draw skeleton on frame for debugging
7. Write a data collection script `scripts/collect_data.py` — opens webcam, records keypoints per sign label, saves to `data/self_recorded/`
8. Write a dataset loader `backend/ml/data/loader.py` — loads INCLUDE dataset from `data/raw/INCLUDE/`, preprocesses into numpy arrays, saves to `data/processed/`
9. Write complete unit tests in `backend/tests/test_pose.py` and `backend/tests/test_buffer.py`
10. Generate `docs/phases/PHASE_1.md` documenting everything implemented
11. Create `requirements.txt` with all pinned dependencies
12. Create `.env.example` with all required environment variables
13. Create `.gitignore` appropriate for this project

---

### Rules you must follow

- Use Python 3.10 type hints on every function signature
- Use Google-style docstrings on every public function and class
- Use `pathlib.Path` for all file paths — never string concatenation
- Use `loguru` for logging — never `print()`
- Use `pydantic` for config validation in `config.py`
- All magic numbers (frame count, confidence threshold, keypoint count) go in config — never hardcoded
- File structure must exactly match ARCHITECTURE.md — no improvisation
- Every function must have a corresponding unit test

---

### Key implementation details

**KeypointExtractor (`backend/pose/extractor.py`):**

```python
# Must extract exactly 130 float32 values per frame:
# - 21 landmarks × left hand × 3 (x, y, z) = 63
# - 21 landmarks × right hand × 3 (x, y, z) = 63
# - 4 pose landmarks (left/right shoulder + wrist) × 1 (x only) = 4
# Total = 130
# If hand not detected, fill with zeros (same shape always)
```

**FrameBuffer (`backend/pose/buffer.py`):**

```python
# Sliding window: window_size=30, stride=15
# When buffer reaches 30 frames: yield np.array of shape (30, 130)
# Advance by 15 frames (keep last 15, add 15 new)
```

**DatasetLoader (`backend/ml/data/loader.py`):**

```python
# INCLUDE dataset structure: data/raw/INCLUDE/{sign_label}/{video_file}.mp4
# For each video: extract all frames → run KeypointExtractor → save as (N_sequences, 30, 130)
# Save processed data: data/processed/isl_sequences.npy and isl_labels.npy
# Print dataset statistics: num_classes, samples_per_class, total_sequences
```

**Data collection script (`scripts/collect_data.py`):**

```python
# CLI tool: python scripts/collect_data.py --sign "hello" --samples 50 --language ISL
# Opens webcam, shows countdown, records 50 samples of the specified sign
# Saves to data/self_recorded/{language}/{sign_label}/sequence_{n}.npy
# Shows live skeleton overlay during recording so user can see their pose
```

---

### What to install (requirements.txt must include these with pinned versions)

```
mediapipe==0.10.14
opencv-python==4.10.0.84
numpy==1.26.4
torch==2.3.1
torchvision==0.18.1
fastapi==0.111.0
uvicorn[standard]==0.30.1
websockets==12.0
python-dotenv==1.0.1
pydantic==2.7.4
pydantic-settings==2.3.4
loguru==0.7.2
pytest==8.2.2
pytest-cov==5.0.0
black==24.4.2
isort==5.13.2
transformers==4.41.2
huggingface-hub==0.23.4
tqdm==4.66.4
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
```

---

### Phase 1 completion checklist — generate this at the end

After implementing everything, output a `docs/phases/PHASE_1.md` file containing:

1. What was implemented (brief description of each module)
2. Dataset statistics (number of signs, sequences, samples per class in INCLUDE)
3. Keypoint extraction performance (average FPS on test video)
4. Unit test results (paste the pytest output)
5. Known limitations or edge cases discovered
6. What Phase 2 will build on top of this

---

### What NOT to do in Phase 1

- Do not implement the ML model yet — that is Phase 2
- Do not implement the FastAPI server yet — that is Phase 4
- Do not implement the frontend — that is Phase 5
- Do not train anything — data collection and preprocessing only
- Do not implement NLP — that is Phase 3

Stay strictly within Phase 1 scope. Output clean, tested, documented code only.

## PROMPT END
