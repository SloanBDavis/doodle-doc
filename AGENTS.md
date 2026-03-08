# AGENTS.md

This file provides guidance to Codex when working in this repository.

## Project Overview

DoodleDoc is a sketch-based search system for handwritten PDF notes. The project is now **accurate-only** and uses **ColQwen2** for both indexing and retrieval.

## Commands

```bash
# Python (use UV, not pip)
uv sync
uv run pytest tests/ -v
uv run ruff check src tests
make dev
make serve

# UI
cd ui && npm install
make ui-dev
make ui-build

# Indexing
make index ROOT=/path/to/pdfs

# Synthetic eval data
uv run doodle-doc synth-build --output data/synth --num-pairs 25
uv run doodle-doc eval data/synth
```

Synthetic generation defaults to Nano Banana 2, `gemini-3.1-flash-image-preview`.
Pass `--model "<model id>"` to `synth-build` if you want a different Google GenAI image model.
Use `--no-clean` to append to an existing synthetic dataset instead of rebuilding it.

## Architecture

**Single retrieval path:**
- Ingest PDFs into a ColQwen2 page index
- Search sketches directly against precomputed ColQwen2 page embeddings

**Key directories:**
- `src/doodle_doc/ingestion/` - PDF rendering and ColQwen2 indexing
- `src/doodle_doc/search/` - ColQwen2 retrieval
- `src/doodle_doc/api/` - FastAPI endpoints
- `src/doodle_doc/synth/` - Synthetic notes/doodle generation and synthetic indexing
- `src/doodle_doc/eval/` - Synthetic evaluation
- `ui/src/components/` - React components

## Code Patterns

- Use UV for Python dependency management
- Keep Python fully typed
- React components use TypeScript with shadcn/ui primitives
- API schemas live in `src/doodle_doc/api/schemas.py`
- The current app does not support legacy fast-mode retrieval paths

## Key Files

- `configs/default.yaml` - runtime configuration
- `src/doodle_doc/ingestion/pipeline.py` - batched ColQwen2 ingestion
- `src/doodle_doc/search/retrieval.py` - ColQwen2 search service
- `src/doodle_doc/synth/pipeline.py` - synthetic dataset build flow
- `src/doodle_doc/eval/runner.py` - synthetic dataset evaluation

## Testing

Tests are in `tests/`. Run:

```bash
uv run pytest tests/ -q
```
