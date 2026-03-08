from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

from doodle_doc.core.config import get_settings, load_settings_from_yaml
from doodle_doc.eval.runner import EvalRunner
from doodle_doc.ingestion.pipeline import IngestionPipeline, IndexingProgress
from doodle_doc.synth.pipeline import SynthConfig, SynthPipeline


def print_progress(progress: IndexingProgress) -> None:
    if progress.pages_total > 0:
        pct = (progress.pages_done / progress.pages_total) * 100
        print(
            f"\r[{progress.status}] {progress.current_doc}: "
            f"{progress.pages_done}/{progress.pages_total} pages ({pct:.1f}%)",
            end="",
            flush=True,
        )


def _load_settings(config_path: str | None):
    settings = get_settings()
    if config_path:
        settings = load_settings_from_yaml(Path(config_path))
    return settings


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.exists():
        print(f"Error: Path does not exist: {root}", file=sys.stderr)
        return 1

    settings = _load_settings(args.config)
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


def cmd_synth_build(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    output_dir = Path(args.output)

    config = SynthConfig(
        output_dir=output_dir,
        num_pairs=args.num_pairs,
        seed=args.seed,
        model=args.model or settings.synth_model,
        prompt_version=settings.synth_prompt_version,
        clean=not args.no_clean,
    )

    print(f"Building synthetic dataset at: {output_dir}")
    print(f"Model: {config.model}")
    print()

    stats = SynthPipeline(settings=settings, config=config).run()

    print()
    print(f"Done! Generated {stats.pages} page/doodle pairs")
    print(f"Indexed pages: {stats.indexed}")
    print(f"Output: {stats.output_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    synth_dir = Path(args.synth_dir)
    if not synth_dir.exists():
        print(f"Error: Synthetic dataset not found: {synth_dir}", file=sys.stderr)
        return 1

    runner = EvalRunner(
        settings=settings,
        synth_dir=synth_dir,
        top_k=args.top_k,
    )
    metrics = runner.run()

    print("\nEvaluation Results")
    print("=" * 40)
    print(f"Recall@1:  {metrics.retrieval.recall_at_1:.3f}")
    print(f"Recall@5:  {metrics.retrieval.recall_at_5:.3f}")
    print(f"Recall@10: {metrics.retrieval.recall_at_10:.3f}")
    print(f"Recall@20: {metrics.retrieval.recall_at_20:.3f}")
    print(f"MRR:       {metrics.retrieval.mrr:.3f}")
    print(f"p50:       {metrics.latency.p50_ms:.0f}ms")
    print(f"p95:       {metrics.latency.p95_ms:.0f}ms")

    if args.check_regression:
        passed, msg = runner.compare_to_baseline(
            metrics,
            threshold=args.regression_threshold,
        )
        status = "PASS" if passed else "FAIL"
        print(f"Regression: {status} - {msg}")

    if args.save_baseline:
        runner.save_as_baseline()
        print("\nBaseline saved")

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
    index_parser.add_argument("--force", "-f", action="store_true", help="Force reindex")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", help="Host to bind to")
    serve_parser.add_argument("--port", "-p", type=int, help="Port to bind to")
    serve_parser.add_argument("--reload", "-r", action="store_true", help="Enable hot reload")

    synth_build_parser = subparsers.add_parser(
        "synth-build",
        help="Generate an eval-ready synthetic dataset and ColQwen index",
    )
    synth_build_parser.add_argument("--output", "-o", default="data/synth", help="Output directory")
    synth_build_parser.add_argument("--num-pairs", "-n", type=int, default=25, help="Number of pairs")
    synth_build_parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    synth_build_parser.add_argument("--model", help="Google GenAI image model to use")
    synth_build_parser.add_argument("--config", "-c", help="Path to config YAML file")
    synth_build_parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Append into the output directory instead of rebuilding it",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a synthetic dataset")
    eval_parser.add_argument("synth_dir", help="Path to synthetic dataset")
    eval_parser.add_argument("--config", "-c", help="Path to config YAML file")
    eval_parser.add_argument("--top-k", "-k", type=int, default=20)
    eval_parser.add_argument("--save-baseline", action="store_true", help="Save results as baseline")
    eval_parser.add_argument("--check-regression", action="store_true", help="Check against baseline")
    eval_parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.05,
        help="Allowed Recall@10 drop before failing regression checks",
    )

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "synth-build":
        return cmd_synth_build(args)
    if args.command == "eval":
        return cmd_eval(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
