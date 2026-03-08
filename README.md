# DoodleDoc

Sketch-based search for handwritten PDF notes using ColQwen2.

## What it does

- Index a folder of PDF notes into a ColQwen2 page index
- Draw a sketch on a canvas to search
- Return matching pages ranked by ColQwen2 late-interaction scoring
- Build synthetic eval datasets with realistic note pages and doodle queries

## Stack

- Backend: FastAPI + Python 3.11+
- Frontend: React + TypeScript + shadcn/ui
- Retrieval: ColQwen2
- Storage: SQLite metadata + ColQwen2 tensor files

## Quick Start

```bash
uv sync
make dev
make ui-dev
```

## Usage

```bash
# Index a folder of PDFs
make index ROOT=/path/to/your/notes

# Build a synthetic dataset and index it
uv run doodle-doc synth-build --output data/synth --num-pairs 25

# Evaluate the synthetic dataset
uv run doodle-doc eval data/synth
```

Pass `--model "<your Nano Banana 2 model id>"` to `synth-build` if you want to use a different Google GenAI image model.

## Project Structure

```text
doodle-doc/
├── src/doodle_doc/api/        # FastAPI routes
├── src/doodle_doc/ingestion/  # PDF processing + ColQwen indexing
├── src/doodle_doc/search/     # ColQwen retrieval
├── src/doodle_doc/synth/      # Synthetic dataset generation + indexing
├── src/doodle_doc/eval/       # Synthetic dataset evaluation
└── ui/                        # React frontend
```

## Requirements

- Python 3.11+
- Node.js 18+
- Enough memory for ColQwen2 during indexing and search

## License

Apache 2.0
