# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DoodleDoc is a sketch-based search system for handwritten PDF notes. Users draw a symbol and the system returns matching pages from their indexed notes.

## Commands

```bash
# Python (use UV, not pip)
uv sync                    # Install dependencies
uv run pytest tests/ -v    # Run tests
make dev                   # Start API server with hot reload
make serve                 # Production server

# UI
cd ui && npm install       # Install UI dependencies
make ui-dev                # Start UI dev server
make ui-build              # Build for production

# Indexing
make index ROOT=/path/to/pdfs

# Linting
make lint                  # Run ruff + mypy
```

## Architecture

**Two-stage retrieval:**
- Stage 1: SigLIP2 embeddings → FAISS search (fast, always runs)
- Stage 2: ColQwen2 reranking (accurate, optional toggle)

**Key directories:**
- `src/doodle_doc/ingestion/` - PDF → embeddings pipeline
- `src/doodle_doc/search/` - Query processing and retrieval
- `src/doodle_doc/api/` - FastAPI endpoints
- `ui/src/components/` - React components (shadcn/ui)

## Code Patterns

- Use UV for all Python dependency management, never pip directly
- All Python code uses type hints (mypy strict mode)
- React components use TypeScript with shadcn/ui primitives
- API responses follow schemas in `src/doodle_doc/api/schemas.py`

## Key Files

- `spec/spec.md` - Detailed specification (read this for implementation details)
- `configs/default.yaml` - All tunable parameters
- `src/doodle_doc/ingestion/embed.py` - SigLIP2 embedder
- `src/doodle_doc/search/rerank.py` - ColQwen2 reranker

## Testing

Tests are in `tests/` mirroring the source structure. Run with `uv run pytest`.
