# SignBridge — System Architecture

## Overview

SignBridge is a real-time, bidirectional sign language interpreter supporting ISL (Indian Sign Language) and NGT (Nederlandse Gebarentaal / Dutch Sign Language). The system is designed as a production-ready, fully decoupled application with a Python ML backend and a React TypeScript frontend communicating over WebSocket for real-time inference.

---

## High-level system diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Webcam feed │    │  Sentence UI │    │ Sign animator│  │
│  │  + skeleton  │    │  (live build)│    │ (reverse mode│  │
│  │  canvas      │    │              │    │              │  │
│  └──────┬───────┘    └──────▲───────┘    └──────▲───────┘  │
│         │                   │                   │           │
│         └──────────WebSocket (ws://localhost:8000/ws)───────┘
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                       BACKEND (FastAPI)                      │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  Pose layer │  │   ML layer   │  │     NLP layer      │ │
│  │  MediaPipe  │→ │  CNN + LSTM  │→ │IndicBERT / BERTje  │ │
│  │  Holistic   │  │  classifier  │  │sentence constructor│ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│                                              │               │
│                         ┌────────────────────▼─────────┐    │
│                         │       Output layer            │    │
│                         │  TTS + Display + Animation    │    │
│                         └───────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## Layer-by-layer breakdown

### Layer 1 — Pose extraction (`backend/pose/`)

**Technology:** MediaPipe Holistic  
**Input:** Raw BGR video frame (H × W × 3)  
**Output:** Numpy array of shape (130,) — flattened keypoints

MediaPipe extracts three sets of landmarks per frame:
- 21 hand landmarks × 2 hands × 3 coordinates = 126 values
- 4 additional pose landmark values (shoulders, wrists for context)
- Total: 130 float32 values per frame

This runs fully on CPU, at 30fps on a modern laptop. No GPU required at this layer.

```
frame (H×W×3) → MediaPipe Holistic → 130 keypoints (x, y, z per landmark)
```

Key module: `backend/pose/extractor.py` → `KeypointExtractor` class

---

### Layer 2 — Frame buffer (`backend/pose/buffer.py`)

**Technology:** Pure Python / NumPy  
**Input:** Stream of (130,) keypoint vectors  
**Output:** Tensor of shape (30, 130) when window is full

A sliding window of 30 frames is maintained. When the buffer is full, it fires the sequence to the ML layer. The window advances by 15 frames (50% overlap) to catch signs that start mid-window.

```
keypoint_t0, keypoint_t1, ... keypoint_t29 → [30 × 130] tensor
```

---

### Layer 3 — ML model (`backend/ml/`)

**Technology:** PyTorch  
**Architecture:** 1D CNN + LSTM hybrid

```
Input: [batch, 30, 130]
         ↓
1D CNN (across feature dim per timestep)
  Conv1D(130 → 64, kernel=3) → BatchNorm → ReLU
  Conv1D(64 → 128, kernel=3) → BatchNorm → ReLU
  → Output: [batch, 30, 128]
         ↓
LSTM (across time dim)
  LSTM(128, hidden=256, layers=2, dropout=0.3)
  → Take last hidden state: [batch, 256]
         ↓
Classifier head
  Linear(256 → 512) → ReLU → Dropout(0.5)
  Linear(512 → num_classes)
  → Softmax
         ↓
Output: probability distribution over sign vocabulary
```

Two separate model instances are loaded at startup:
- `isl_model.pt` — trained on INCLUDE + ISL-CSLRT + self-recorded ISL signs
- `ngt_model.pt` — trained on Corpus NGT

Active model is swapped based on language toggle from frontend.

---

### Layer 4 — Word accumulator (`backend/ml/inference/accumulator.py`)

**Technology:** Pure Python  
**Input:** Stream of (sign_label, confidence) tuples  
**Output:** Committed word list

Rules:
- Confidence threshold: 0.85 (configurable in `config.yaml`)
- A sign is committed when it appears in 2 consecutive windows above threshold
- Pause detection: if no sign is detected for 1.5 seconds, sentence is finalised
- Duplicate suppression: same sign cannot be committed twice consecutively

---

### Layer 5 — NLP sentence construction (`backend/nlp/`)

**Technology:** HuggingFace Transformers  
**Models:**
- ISL → IndicBERT (ai4bharat/indic-bert) fine-tuned on ISL grammar corpus
- NGT → BERTje (GroNLP/bert-base-dutch-cased) fine-tuned on NGT grammar corpus

**Input:** Raw word list following sign language grammar order  
**Output:** Grammatically correct natural language sentence

ISL and NGT have their own grammar rules (topic-comment structure, different word order than English/Dutch). The fine-tuned transformers handle this sequence-to-sequence correction.

---

### Layer 6 — Output (`backend/tts/` and `backend/api/`)

Three parallel outputs sent back to frontend via WebSocket:

1. **Display text** — constructed sentence string
2. **Audio** — gTTS generates MP3 bytes, streamed to frontend
3. **Sign animation data** — for reverse mode, JSON array of keypoint sequences

---

### Layer 7 — Frontend (`frontend/`)

**Technology:** React 18, TypeScript, Tailwind CSS, Zustand

**Key components:**
- `CameraView` — getUserMedia webcam feed + canvas skeleton overlay drawn with requestAnimationFrame
- `SentencePanel` — live word-by-word sentence construction with typewriter animation
- `LanguageToggle` — ISL / NGT switch, fires language change event to backend
- `ReverseMode` — text input → sign animation player
- `SkeletonOverlay` — draws MediaPipe keypoints on canvas using frontend-side MediaPipe (for low-latency visual feedback, separate from backend processing)

**WebSocket message schema:**

Frontend → Backend (every frame):
```json
{ "type": "frame", "data": "<base64 JPEG>", "language": "ISL" }
```

Backend → Frontend (on recognition):
```json
{
  "type": "recognition",
  "word": "water",
  "confidence": 0.94,
  "sentence": "I want water",
  "audio_b64": "<base64 MP3>"
}
```

---

## Data flow — forward mode (signing → text)

```
1. Webcam captures frame at 30fps
2. Frontend sends frame to backend via WebSocket
3. Backend pose extractor runs MediaPipe → 130 keypoints
4. Frame buffer accumulates 30 frames → [30×130] tensor
5. CNN+LSTM model infers sign label + confidence
6. Accumulator commits word when threshold met
7. NLP layer constructs sentence from word list
8. Backend sends {word, sentence, audio} back to frontend
9. Frontend updates sentence panel + plays audio
```

## Data flow — reverse mode (text → sign animation)

```
1. User types sentence in reverse mode panel
2. Frontend sends sentence to REST endpoint POST /api/v1/reverse
3. Backend NLP layer tokenises sentence into sign vocabulary words
4. Keypoint library looks up reference sequences for each word
5. Backend returns ordered array of keypoint sequences
6. Frontend animates skeleton frame-by-frame on canvas
```

---

## Deployment architecture (future)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Vercel    │     │  Railway /   │     │  Hugging     │
│  (Frontend) │────▶│  Render      │────▶│  Face Hub    │
│             │     │  (FastAPI)   │     │  (Models)    │
└─────────────┘     └──────────────┘     └──────────────┘
```

Frontend is fully static — deployable to Vercel/Netlify with zero config.  
Backend is containerised via Docker — deployable to any cloud with GPU support.  
Model weights hosted on HuggingFace Hub, pulled at container startup.

---

## Technology decisions log

| Decision | Choice | Reason |
|---|---|---|
| ML framework | PyTorch | More flexible than Keras for custom architectures |
| Pose library | MediaPipe | Runs on CPU, 30fps, no setup overhead |
| Backend framework | FastAPI | Native async, WebSocket support, auto OpenAPI docs |
| Frontend state | Zustand | Simpler than Redux, enough for this scale |
| NLP — ISL | IndicBERT | Best multilingual Indian language model available |
| NLP — NGT | BERTje | Best Dutch BERT model, Radboud University |
| Styling | Tailwind | Utility-first, consistent design system |
| Model format | PyTorch .pt | Faster loading than ONNX for development phase |
