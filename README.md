# DoodleDoc

Sketch-based search for handwritten PDF notes. Draw a symbol, find where you wrote it.

## What it does

- Index a folder of PDF notes (Notability exports, scanned handwriting, etc.)
- Draw a sketch on a canvas to search
- Get back matching pages ranked by similarity
- Optional "High Accuracy" mode uses a stronger model for reranking

## Architecture

**Two-stage retrieval:**
1. **Fast search** (~300ms): SigLIP2 embeddings + FAISS vector search
2. **Accurate search** (~3s, optional): ColQwen2 reranks top results

**Stack:**
- Backend: FastAPI + Python 3.11+
- Frontend: React + TypeScript + shadcn/ui
- ML: PyTorch with MPS (Apple Silicon) support
- Index: FAISS for vectors, BM25 for text

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4) with 16GB+ RAM
- 24GB recommended for ColQwen2 reranking
- Python 3.11+
- Node.js 18+

## Quick Start

```bash
# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start the API server
make dev

# In another terminal, start the UI
make ui-dev
```

## Usage

```bash
# Index a folder of PDFs
make index ROOT=/path/to/your/notes

# Run the server
make serve

# Open http://localhost:8000
```

## Project Structure

```
doodle-doc/
├── src/doodle_doc/    # Python backend
│   ├── api/           # FastAPI routes
│   ├── ingestion/     # PDF processing pipeline
│   └── search/        # Retrieval + reranking
├── ui/                # React frontend
|── configs/           # YAML configuration
```
## License

Apache 2.0
