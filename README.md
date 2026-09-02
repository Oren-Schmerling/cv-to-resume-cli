## Offline / Local Model

- Small language model (SLM) hosted locally via **vLLM**.
- Structured outputs (`response_format` `json_schema`) enforces valid JSON output at decode time.
- No external API calls — entire pipeline (ranking, rendering, compiling) runs offline.
- `tectonic` used for PDF compilation (self-contained, no TeX Live install needed).

## System Design Choices Made

- LLM only ranks/selects, generates nothing.
- Python assembles, validates, and compiles.
- CV entries authored so each entry = exactly 1 rendered line → No LLM-side line/math logic

## Phase 2 Usage: LLM Matching (library API)

With the vLLM server running (above), `select_entries` sends the CV entries
and a job description to the model and returns `{section: [ids]}` over all
configured sections (sections with no CV entries come back as `[]`):

```python
from pathlib import Path

from cv_to_resume_cli import (
    MatcherError,
    load_and_cross_validate,
    select_entries,
)

entries, config = load_and_cross_validate(
    Path("data/cv.json"), Path("data/section_config.json")
)

try:
    selection = select_entries(
        "Senior platform engineer role. Kubernetes, Terraform, Python...",
        entries,
        config,
    )
except MatcherError as exc:
    raise SystemExit(f"matching failed: {exc}")

print(selection)
# {"experience": ["exp-004"], "education": ["edu-003"], "skills": ["skill-004"]}
```

Behavior:

- Structured outputs enforce the decoding schema server-side: per-section
  `enum` of valid CV ids, `maxItems` = `max_entries`, `uniqueItems`.
- The client re-validates every response (ids exist, no duplicates, caps,
  exact keys) and retries malformed/invalid output up to 3 attempts total.
- Connection/HTTP errors are not retried — they raise `MatcherError` at once.
- If the configured model name returns 404, one `GET /v1/models` discovery
  call runs; the single served model is used, 0 or ≥2 models raise
  `MatcherError`.
- Host/port/model dir come from `Settings` (`VLLM_HOST`, `VLLM_PORT`,
  `HF_MODEL_LOCAL_DIR` in `.env`). CLI wiring lands in Phase 6.
- Opt-in live test: `LIVE_VLLM_TESTS=1 uv run pytest tests/` (skipped
  otherwise; structural asserts only, no golden ids until Phase 7).

## Dependencies & Prerequisites

### Runtime

- Python 3.10+
- vLLM (local model serving)
- Structured outputs: vLLM `response_format` `json_schema` (backend auto: xgrammar → guidance → outlines)
- Local SLM weights (GPU recommended)
- Jinja2 (custom delimiters for LaTeX)
- `tectonic` (PDF compilation, self-contained — no TeX Live)
- `jsonschema` (cv.json / section_config.json validation)

### Development / Testing

- pytest
- Golden-file fixtures (job description → expected IDs)
- Sample `cv.json` + `section_config.json` for testing
- LaTeX template files (`.tex` w/ custom Jinja2 delimiters)

### System / Environment

- GPU (recommended for vLLM inference)
- No internet required at runtime (fully offline)
- CWD write access (PDF output)

## Setup (uv)

- Install [uv](https://docs.astral.sh/uv/).
- `uv venv` — create virtualenv.
- `uv sync` — install from lockfile (`uv.lock`).
- `uv run <script>` — run CLI/scripts.

And use the following if you ever need to add deps:

- `uv add <package>` — add deps (never `pip install`).
- `uv lock` — regenerate lockfile after dep changes.

### GPU vs CPU (auto-detected)

This project runs on GPU or CPU-only machines. `uv` auto-selects the right PyTorch/vLLM backend:

```bash
uv pip install vllm --torch-backend=auto
```

- `auto` detects an NVIDIA driver and installs the matching CUDA build.
- On a machine with no GPU, it falls back to CPU wheels automatically.
- Manual CPU-only install (if `auto` fails):

```bash
  uv add vllm --index https://download.pytorch.org/whl/cpu
  uv sync
```

### Other requirements

1. **GPU drivers / CUDA (GPU machines only)**

```bash
   nvidia-smi # verify driver + CUDA version
```

Skip this step on CPU-only machines — vLLM CPU backend needs no CUDA.

2. **Hugging Face CLI (model download)**

```bash
   uv add "huggingface_hub[cli]"

   # login only if model is gated/private
   uv run huggingface-cli login

   # download SLM weights to HF_MODEL_LOCAL_DIR
   uv run huggingface-cli download $HF_MODEL_ID --local-dir $HF_MODEL_LOCAL_DIR
```

Set `HF_MODEL_ID`, `HF_MODEL_LOCAL_DIR`, and `HF_TOKEN` (if needed) in `.env`.

3. **vLLM model server**

```bash
   uv run vllm serve $HF_MODEL_LOCAL_DIR \
     --host $VLLM_HOST --port $VLLM_PORT
```

   Optional: pin the structured-outputs backend (default `auto`, which falls
   back xgrammar → guidance → outlines per request with validation):

```bash
   uv run vllm serve $HF_MODEL_LOCAL_DIR \
     --structured-outputs-config.backend outlines \
     --host $VLLM_HOST --port $VLLM_PORT
```

4. **tectonic (LaTeX compiler)**

```bash

   # macOS

   brew install tectonic

   # Linux

   curl --proto '=https' -fsSL https://drop-sh.fullyjustified.net | sh

   # Windows (PowerShell)

   winget install --id TectonicTypesetting.Tectonic

   tectonic --version # verify
```

5. **Project data files**

```bash
   touch cv.json section_config.json

   # populate per schema in docs; place LaTeX template in expected path

```

6. **Environment file**

```bash
   cp .env.example .env
   # edit .env: paths, HF_MODEL_ID, HF_TOKEN, vLLM host/port
```

7. **Verify pipeline**

```bash
   uv run resume_cli.py --dry-run
```

## File structure

```
resume-cli/
├── .env
├── .env.example
├── pyproject.toml
├── uv.lock
├── README.md
├── resume_cli.py
├── data/
│   ├── cv.json
│   ├── cv.example.json
│   └── section_config.json
├── schemas/
│   ├── cv.schema.json
│   └── section_config.schema.json
├── templates/
│   └── resume.tex.jinja
└── tests/
    ├── test_data_validation.py
    ├── test_selection.py
    ├── test_latex_escaping.py
    ├── test_golden_files.py
    └── fixtures/
        ├── sample_cv.json
        ├── sample_section_config.json
        └── golden/
```
