# DoodleDoc

Sketch-based search for handwritten PDF notes. Draw a symbol, find where you wrote it.

## What it does

- Index a folder of PDF notes (Notability exports, scanned handwriting, etc.)
- Draw a sketch on a canvas to search
- Get back matching pages ranked by similarity
- Optional "High Accuracy" mode uses a stronger model for reranking

## Updates
- Just started this today, looking to have it working sometime in January (Dec 17, 2025)
- Search with sigLIP2 is working and there is a basic UI. Ingestion and search both work (Dec 19, 2025)
- Added reranking with ColQwen2. It is slow right now but the accuracy is higher, doodles require less features to return an accurate result (Dec 21, 2025)
- Switched to embedding with ColQwen2 at index time, this has increased accuracy by 50% on my sanity set with simple graphs (Dec 22, 2025)

## Architecture

**Two-pronged retrieval:**
1. **Fast search** : SigLIP2 embeddings + FAISS vector search
2. **Accurate search** : ColQwen2 for more accurate results

**Note** Used to use SigLIP2 for initial search and then ColQwen2 for reranking

**Stack:**
- Backend: FastAPI + Python 3.11+
- Frontend: React + TypeScript + shadcn/ui
- ML: PyTorch
- Index: FAISS for vectors, BM25 for text

## Requirements

- 24GB RAM recommended for ColQwen2 reranking
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
