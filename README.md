# SignBridge

Real-time, bidirectional sign language interpreter supporting **ISL** (Indian Sign Language) and **NGT** (Dutch Sign Language).

## Stack

| Layer | Technology |
|---|---|
| Pose extraction | MediaPipe Holistic (Tasks API v1.0+) |
| ML model | PyTorch — 1D CNN + LSTM |
| Backend | FastAPI + WebSocket |
| Frontend | React 18 + TypeScript + Tailwind |
| NLP | IndicBERT (ISL) / BERTje (NGT) |

## Datasets

| Dataset | Language | Type |
|---|---|---|
| INCLUDE | ISL | Isolated signs — 263 classes, 4292 videos |
| ISLVT | ISL | Sentence videos + Marathi/English gloss |
| Indian Sign Language_Dataset | ISL | Isolated sign videos |
| NGT HoReCo 1.2 | NGT | Continuous signing — 297 videos + annotation |

Datasets live in `Datasets/` at repo root and are gitignored.

## Phase 1 Setup

### 1. Activate conda

```powershell
(C:\Users\Deep\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate base)
```

### 2. Install dependencies

```powershell
cd signbridge
pip install -r requirements.txt
```

### 3. Download the MediaPipe holistic model

```powershell
Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task -OutFile signbridge/backend/models/holistic_landmarker.task -UseBasicParsing
```

### 4. Run tests

```powershell
cd signbridge
pytest backend/tests/ -v
```

### 5. Collect self-recorded signs

```powershell
python scripts/collect_data.py --sign "hello" --samples 50 --language ISL
```

### 6. Process datasets

```python
from backend.ml.data.loader import DatasetLoader

with DatasetLoader() as loader:
    loader.process_include()
    loader.process_islvt()
    loader.process_isl_dataset()
    loader.process_ngt_horeco()
```

## Repository Structure

```
ISL_NGT-Sign-Language-Interpreter/
├── Datasets/                          # Raw datasets (gitignored)
│   ├── INCLUDE/
│   ├── Indian Sign Language Video.../
│   ├── Indian Sign Language_Dataset/
│   └── NGT_HoReCo_1.2/
├── MDs/                               # Project design documents
│   ├── RULES.md
│   ├── ARCHITECTURE.md
│   └── DESIGN.md
├── signbridge/                        # All source code
│   ├── backend/
│   │   ├── core/                      # config.py, logger.py
│   │   ├── pose/                      # extractor.py, buffer.py, visualizer.py
│   │   ├── ml/data/                   # loader.py
│   │   ├── models/                    # .task / .pt weights (gitignored)
│   │   └── tests/
│   ├── data/processed/                # numpy arrays (gitignored)
│   ├── data/self_recorded/            # webcam recordings (gitignored)
│   ├── docs/phases/PHASE_1.md
│   ├── scripts/collect_data.py
│   ├── requirements.txt
│   └── pyproject.toml
├── .gitignore
├── INIT_PROMPT.md
└── README.md
```

## Phase Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Scaffold + pose extraction pipeline | ✅ Complete |
| 2 | CNN + LSTM model training | 🔜 |
| 3 | NLP sentence construction | 🔜 |
| 4 | FastAPI + WebSocket backend | 🔜 |
| 5 | React frontend | 🔜 |
