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
from doodle_doc.eval.runner import EvalRunner


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


def cmd_eval(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.config:
        settings = load_settings_from_yaml(Path(args.config))

    modes = None
    if args.mode != "both":
        modes = [args.mode]

    runner = EvalRunner(
        settings=settings,
        num_queries=args.num_queries,
        seed=args.seed,
        regenerate=args.regenerate,
    )

    results = runner.run(modes=modes)

    print("\nEvaluation Results")
    print("=" * 40)

    for mode, metrics in results.items():
        mode_label = "Fast (SigLIP2)" if mode == "fast" else "Accurate (ColQwen2)"
        print(f"\n{mode_label}:")
        print(f"  Recall@1:  {metrics.retrieval.recall_at_1:.3f}")
        print(f"  Recall@5:  {metrics.retrieval.recall_at_5:.3f}")
        print(f"  Recall@10: {metrics.retrieval.recall_at_10:.3f}")
        print(f"  Recall@20: {metrics.retrieval.recall_at_20:.3f}")
        print(f"  MRR:       {metrics.retrieval.mrr:.3f}")
        print(f"  p50:       {metrics.latency.p50_ms:.0f}ms")
        print(f"  p95:       {metrics.latency.p95_ms:.0f}ms")

        if args.check_regression:
            passed, msg = runner.compare_to_baseline(
                metrics, mode, threshold=args.regression_threshold
            )
            status = "PASS" if passed else "FAIL"
            print(f"  Regression: {status} - {msg}")

    if args.save_baseline:
        for mode in results:
            runner.save_as_baseline(mode)
        print(f"\nBaseline saved for: {', '.join(results.keys())}")

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

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--config", "-c", help="Path to config YAML file")
    eval_parser.add_argument("--mode", choices=["fast", "accurate", "both"], default="both")
    eval_parser.add_argument("--regenerate", action="store_true", help="Regenerate pseudo-queries")
    eval_parser.add_argument("--num-queries", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--save-baseline", action="store_true", help="Save results as baseline")
    eval_parser.add_argument("--check-regression", action="store_true", help="Check against baseline")
    eval_parser.add_argument("--regression-threshold", type=float, default=0.05)

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "eval":
        return cmd_eval(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
