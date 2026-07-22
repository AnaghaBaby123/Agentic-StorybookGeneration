# AGENTIC AI Storybook Generator

An automated pipeline that turns a single theme into a fully illustrated children's storybook PDF. It writes the story, breaks it into pages, generates matching illustrations, keeps a human in the loop at key checkpoints, and composes everything into a final PDF.
## How it works

The system is split into two clearly separated layers:

- **MCP server** (`mcpserver.py`) — owns all content generation. Every LLM call (writing the story, evaluating it, writing image prompts) and every image generation call lives here as an MCP tool. Nothing about orchestration or state lives in this layer.
- **LangGraph pipeline** (`storybook_.py`) — owns orchestration only. It tracks pipeline state, routes between nodes, and pauses for human review at two checkpoints. It never calls an LLM or image model directly — it calls MCP tools.

![Pipeline flowchart](docs/flowchart.svg)


### Models

| Component | Model | 
|---|---|
| Story & prompt generation | Qwen3-8B-AWQ 
| Illustrations | FLUX .1 Schnell|

Both models run locally as subprocesses — no external API calls for generation.

## Requirements

- 30 GB VRAM or Kaggle notebook with **2x Tesla T4** GPUs enabled (was using kaggle for VRAM services)
- Python 3.12
- CUDA 12.8, PyTorch 2.10.0+cu128
- A Hugging Face account + access token (`HF_TOKEN`), for pulling Qwen3-8B-AWQ and SDXL weights

### Python dependencies

```bash
pip install requirements.txt
```

## Setup

### 1. Set your Hugging Face token

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

The token must also be explicitly passed into the MCP server subprocess's environment (see `env=os.environ.copy()` in `storybook.py`) — implicit inheritance doesn't reliably carry CUDA or HF env vars to subprocesses in Kaggle.

### 2. Download the Noto Sans font (required for PDF text)

The PDF composer uses a Unicode font instead of the default PDF-builtin Helvetica, because Helvetica only supports Latin-1 and breaks on smart quotes, em dashes, and other characters LLM-generated text commonly includes.

Download **Noto Sans** and place the `.ttf` file at the path your pipeline expects (default: `fonts/NotoSans-Regular.ttf` in the project root):

```bash
mkdir -p fonts
curl -L -o fonts/NotoSans-Regular.ttf \
  "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf"
```

If you also want bold titles/headers to render correctly, grab the bold weight too:

```bash
curl -L -o fonts/NotoSans-Bold.ttf \
  "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Bold.ttf"
```

Then register it in your `compose_pdf` step (ReportLab example):

```python
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("NotoSans", "fonts/NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSans-Bold", "fonts/NotoSans-Bold.ttf"))

canvas.setFont("NotoSans", 12)
```

On Kaggle, run the pipeline's font check first if you're unsure whether a system font is already available:

```bash
fc-list | grep -i noto
```

If nothing is found, use the `curl` commands above rather than relying on the system font cache.

### 3. Start vLLM (GPU 1)

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --attention-backend TRITON_ATTN \
  --enforce-eager
```

> T4 GPUs don't support FlashInfer JIT (missing `libcuda.so` linker stub) — `TRITON_ATTN` and `--enforce-eager` are required, not optional, on this hardware.

### 4. Run the pipeline

```
python storybook.py
```

The run will pause twice for human input:
1. **Story review** — approve the story or request a rewrite with feedback.
2. **Page review** — approve each page's illustration or request a regenerate.

## Project structure

```
.
├── mcpserver.py           # MCP server: all LLM + image generation tools
├── storybook_pipeline.py  # LangGraph orchestration + human-in-the-loop
├── docs/
│   └── flowchart.svg      # pipeline flow diagram (referenced in README)
├── fonts/
│   └── NotoSans-Regular.ttf
├── output/                # generated PDFs land here
└── README.md
```

## Known constraints

- Character consistency is prompt-based only (no IP-Adapter/LoRA), so expect thematic consistency across pages rather than pixel-identical characters.
- `print()` statements anywhere in `mcpserver.py` will corrupt the MCP stdio channel — all logging must go through the logger, never stdout.

