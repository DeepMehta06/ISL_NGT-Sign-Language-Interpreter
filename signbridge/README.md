# SignBridge

Real-time, bidirectional sign langue interpreter supporting **ISL** (Indian Sign Language) and **NGT** (Dutch Sign Language).

## Stack

| Layer           | Technology                       |
| --------------- | -------------------------------- |
| Pose extraction | MediaPipe Holistic               |
| ML model        | PyTorch — 1D CNN + LSTM         |
| Backend         | FastAPI + WebSocket              |
| Frontend        | React 18 + TypeScript + Tailwind |
| NLP             | IndicBERT (ISL) / BERTje (NGT)   |

## Datasets

| Dataset                      | Language | Type                                       | Location                                                                           |
| ---------------------------- | -------- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| INCLUDE                      | ISL      | Isolated signs — 263 classes, 4292 videos | `../Datasets/INCLUDE/`                                                           |
| ISLVT                        | ISL      | Sentence videos + gloss Excel              | `../Datasets/Indian Sign Language Video and Text dataset for sentences (ISLVT)/` |
| Indian Sign Language_Dataset | ISL      | Compressed sign videos                     | `../Datasets/Indian Sign Language_Dataset/ISL.zip`                               |
| NGT HoReCo 1.2               | NGT      | Continuous signing — 297 videos + Excel   | `../Datasets/NGT_HoReCo_1.2/`                                                    |

## Phase 1 Setup

### 1. Activate conda environment

```powershell
(C:\Users\Deep\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate base)
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run tests

```powershell
pytest backend/tests/ -v
```

### 4. Collect self-recorded signs

```powershell
python scripts/collect_data.py --sign "hello" --samples 50 --language ISL
```

### 5. Process datasets

```python
from backend.ml.data.loader import DatasetLoader

with DatasetLoader() as loader:
    loader.process_include()       # INCLUDE ISL isolated signs
    loader.process_islvt()         # ISLVT ISL sentence videos
    loader.process_isl_dataset()   # Indian Sign Language_Dataset
    loader.process_ngt_horeco()    # NGT HoReCo 1.2
```

## Project Structure

```
signbridge/
├── backend/
│   ├── api/              # FastAPI routes (Phase 4)
│   ├── core/             # config.py, logger.py
│   ├── ml/
│   │   ├── data/         # loader.py
│   │   ├── training/     # Phase 2
│   │   ├── inference/    # Phase 2
│   │   └── evaluation/   # Phase 2
│   ├── models/           # .pt weights (not committed)
│   ├── nlp/              # Phase 3
│   ├── pose/             # extractor.py, buffer.py, visualizer.py
│   ├── tests/
│   └── tts/              # Phase 4
├── data/
│   ├── processed/        # numpy arrays output
│   └── self_recorded/    # webcam-recorded sequences
├── docs/phases/
├── notebooks/
├── scripts/
│   └── collect_data.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Phase Roadmap

| Phase | Scope                               |
| ----- | ----------------------------------- |
| 1 ✅  | Scaffold + pose extraction pipeline |
| 2     | CNN + LSTM model training           |
| 3     | NLP sentence construction           |
| 4     | FastAPI + WebSocket backend         |
| 5     | React frontend                      |
