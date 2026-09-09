# SignBridge — Project Rules & conventions

> Read this before touching any file. Every contributor (including AI agents) must follow these rules without exception.

---

## 1. Environment setup — run this first, every time

Before running ANY Python command in this project, execute the following in PowerShell:

```powershell
(C:\Users\Deep\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate base)
```

Never run Python scripts outside this activated environment. Dependencies are pinned to this environment only.

---

## 2. Project identity

| Field               | Value                                                 |
| ------------------- | ----------------------------------------------------- |
| Project name        | SignBridge                                            |
| Version             | 0.1.0                                                 |
| Languages supported | ISL (Indian Sign Language), NGT (Dutch Sign Language) |
| Primary stack       | Python 3.10, FastAPI, React 18, TypeScript            |
| ML framework        | PyTorch                                               |
| Pose extraction     | MediaPipe Holistic                                    |

---

## 3. Repository structure — never deviate from this

```
signbridge/
├── backend/
│   ├── api/                  # FastAPI route handlers
│   ├── core/                 # Config, logging, constants
│   ├── models/               # Trained model files (.pt) — never commit large files
│   ├── ml/
│   │   ├── data/             # Dataset loaders, preprocessors
│   │   ├── training/         # Training scripts, loss functions
│   │   ├── inference/        # Real-time inference pipeline
│   │   └── evaluation/       # Metrics, confusion matrix, reports
│   ├── pose/                 # MediaPipe extraction logic
│   ├── nlp/                  # Sentence construction, transformer fine-tuning
│   ├── tts/                  # Text-to-speech wrappers
│   ├── tests/                # All backend tests (pytest)
│   └── main.py               # FastAPI entry point
├── frontend/
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/            # Page-level components
│   │   ├── hooks/            # Custom React hooks (useWebSocket, useCamera)
│   │   ├── store/            # Zustand global state
│   │   ├── services/         # API/WebSocket service layer
│   │   ├── types/            # TypeScript interfaces
│   │   └── styles/           # Global styles, design tokens
│   ├── public/
│   └── package.json
├── data/
│   ├── raw/                  # Original downloaded datasets — never modify
│   │   ├── INCLUDE/
│   │   ├── ISL-CSLRT/
│   │   ├── corpus-ngt/
│   │   └── signbank-nl/
│   ├── processed/            # Preprocessed numpy arrays, ready for training
│   └── self_recorded/        # Your own recorded signs
├── notebooks/                # Exploration and EDA only — no production code here
├── docs/                     # All documentation
│   ├── RULES.md              # This file
│   ├── ARCHITECTURE.md
│   ├── DESIGN.md
│   └── phases/               # Phase-by-phase documentation
│       ├── PHASE_1.md
│       ├── PHASE_2.md
│       └── ...
├── scripts/                  # One-off utility scripts (data download, conversion)
├── .env.example              # Environment variable template — never commit .env
├── .gitignore
├── requirements.txt          # Backend Python deps (pinned versions)
└── README.md
```

---

## 4. Phase gate rules — mandatory before moving forward

Every phase must pass ALL of the following before the next phase begins:

### Documentation gate

- [ ] Phase markdown file written in `docs/phases/PHASE_N.md`
- [ ] All functions have docstrings (Google style)
- [ ] README updated with new setup steps if any

### Testing gate

- [ ] Unit tests written for every new module
- [ ] All tests passing: `pytest backend/tests/ -v`
- [ ] Test coverage above 70% for new code: `pytest --cov`

### Review gate

- [ ] No hardcoded paths — all paths via config
- [ ] No secrets in code — all via `.env`
- [ ] No `print()` statements — use logger
- [ ] Code formatted: `black backend/` and `isort backend/`
- [ ] No unused imports

Only after all boxes are checked do you move to the next phase.

---

## 5. Coding conventions — Python (backend)

- Python version: 3.10 strictly
- Formatter: `black` (line length 88)
- Import sorter: `isort`
- Type hints: mandatory on all function signatures
- Docstrings: Google style, mandatory on all public functions and classes
- Logging: use `backend/core/logger.py` — never use `print()`
- Config: all constants live in `backend/core/config.py` — no magic numbers in code
- Error handling: always raise typed exceptions, never bare `except:`

```python
# CORRECT
def extract_keypoints(frame: np.ndarray) -> np.ndarray:
    """Extract MediaPipe keypoints from a single video frame.

    Args:
        frame: BGR image array of shape (H, W, 3).

    Returns:
        Flattened keypoint array of shape (130,).
    """
    ...

# WRONG — never do this
def extract(f):
    ...
```

---

## 6. Coding conventions — TypeScript (frontend)

- Framework: React 18 with TypeScript strict mode
- State: Zustand (no Redux)
- Styling: Tailwind CSS only — no inline styles, no CSS modules
- Component naming: PascalCase for components, camelCase for hooks
- All props must have TypeScript interfaces defined in `src/types/`
- No `any` type — ever
- Hooks must start with `use`
- Every component file exports exactly one default component

---

## 7. ML model rules

- All model architectures defined in `backend/ml/models/` as PyTorch `nn.Module` classes
- Model weights saved to `backend/models/` as `.pt` files — never `.pkl`
- Every model class must implement `forward()` with typed inputs and outputs
- Training configs (learning rate, epochs, batch size) live in YAML files inside `backend/ml/training/configs/` — never hardcoded
- Every training run logs to `runs/` directory with timestamp
- Model evaluation report must be generated before any model is used in inference

---

## 8. API conventions

- All endpoints versioned under `/api/v1/`
- WebSocket endpoint for real-time inference: `ws://localhost:8000/ws/inference`
- All REST responses follow this shape:

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

- HTTP status codes used correctly — never return 200 for an error
- All endpoints documented with FastAPI's built-in OpenAPI

---

## 9. Git conventions

- Branch naming: `feat/phase-1-pose-extraction`, `fix/lstm-training-bug`
- Commit messages: conventional commits format
  - `feat: add MediaPipe keypoint extraction`
  - `fix: correct LSTM sequence length mismatch`
  - `docs: update PHASE_1.md with test results`
  - `test: add unit tests for frame buffer`
- Never commit directly to `main`
- Every feature branch gets a PR with a short description

---

## 10. What AI agents must never do

- Never modify files in `data/raw/` — those are source of truth
- Never hardcode file paths — always use `pathlib.Path` and config
- Never skip the phase gate checklist
- Never write production logic in notebooks
- Never commit `.env`, model weights, or dataset files to git
- Never use `any` type in TypeScript
- Never bypass the conda environment activation step
- Never write unnecesarry lots of comments

---

## 11. Documentation diagrams � mandatory Mermaid

All architecture diagrams in any `docs/` or `PHASE_N.md` file **must** use Mermaid syntax. Plain ASCII box art is not allowed.

Rules for Mermaid diagrams:
- Use `flowchart TD` for top-down data flow (model architecture, pipeline stages)
- Use `graph LR` for horizontal dependency graphs
- Use `sequenceDiagram` for request/response or real-time event flows
- Every node must include the **tensor shape** or **data type** relevant to that stage
- Layer nodes must show: layer type, key parameters (channels, kernel size, etc.)
- Use subgraphs to group logical blocks (e.g. Spatial Encoder, Temporal Encoder, Classifier Head)
- Node labels containing parentheses or special characters must be quoted: `id["Label (shape)"]`
- Do NOT use HTML tags inside Mermaid node labels

Any AI agent writing architecture documentation that uses ASCII diagrams instead of Mermaid is in violation of this rule and must rewrite the diagram.



---

## 12. AI agent command execution policy

AI agents must **never run commands autonomously** on behalf of the user unless the user explicitly says "run it" or "execute".**

- Always provide the exact command(s) the user should run themselves.
- Do not use un_command to test or verify scripts unless the user has granted explicit permission for that specific command.
- Wasting API credits on command execution that the user intended to run themselves is a violation of this rule.


---

## 13. GPU / CUDA usage — always prefer GPU

- `settings.device` auto-detects CUDA via `torch.cuda.is_available()`. Never hardcode `'cpu'`  or `'cuda'` anywhere in code — always read from `settings.device`.\r
- Training (`Trainer`), inference, and evaluation always move tensors to `settings.device`.\r
- DataLoader `pin_memory=True` when CUDA is available — reduces CPU->GPU transfer latency.\r
- MediaPipe pose extraction is CPU-only (no CUDA support). Do not attempt GPU acceleration there.\r
- To override device manually, set `DEVICE=cpu` in the `.env` file. Never change code for this.\r
- When adding any new PyTorch component, always call `.to(device)` on the model and `.to(self.device)` on all tensors inside training/inference loops.\r

