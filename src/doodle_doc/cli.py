from __future__ import annotations

import os

# Fix OpenMP duplicate library error on macOS
# Must be set before importing torch, faiss, or opencv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

from doodle_doc.core.config import get_settings, load_settings_from_yaml
from doodle_doc.ingestion.pipeline import IngestionPipeline, IndexingProgress


def print_progress(progress: IndexingProgress) -> None:
    if progress.pages_total > 0:
        pct = (progress.pages_done / progress.pages_total) * 100
        print(
            f"\r[{progress.status}] {progress.current_doc}: "
            f"{progress.pages_done}/{progress.pages_total} pages ({pct:.1f}%)",
            end="",
            flush=True,
        )


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.exists():
        print(f"Error: Path does not exist: {root}", file=sys.stderr)
        return 1

    settings = get_settings()
    if args.config:
        settings = load_settings_from_yaml(Path(args.config))

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing PDFs from: {root}")
    print(f"Data directory: {settings.data_dir}")

    pipeline = IngestionPipeline(settings)
    progress = pipeline.run(
        root,
        on_progress=print_progress,
        force_reindex=args.force,
    )

    print()
    print(f"Done! Indexed {progress.docs_done} documents, {progress.pages_done} pages.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host or "127.0.0.1"
    port = args.port or 8000

    print(f"Starting DoodleDoc API at http://{host}:{port}")
    uvicorn.run(
        "doodle_doc.api.main:app",
        host=host,
        port=port,
        reload=args.reload,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="doodle-doc",
        description="Sketch-based search for handwritten PDF notes",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index PDF files")
    index_parser.add_argument("path", help="Path to folder containing PDFs")
    index_parser.add_argument("--config", "-c", help="Path to config YAML file")
    index_parser.add_argument("--force", "-f", action="store_true", help="Force reindex all files")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", help="Host to bind to")
    serve_parser.add_argument("--port", "-p", type=int, help="Port to bind to")
    serve_parser.add_argument("--reload", "-r", action="store_true", help="Enable hot reload")

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    elif args.command == "serve":
        return cmd_serve(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
