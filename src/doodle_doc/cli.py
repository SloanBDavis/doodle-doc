from __future__ import annotations

import os

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


def cmd_synth_generate(args: argparse.Namespace) -> int:
    from doodle_doc.synth.pipeline import SynthPipeline, SynthConfig

    config = SynthConfig(
        output_dir=Path(args.output),
        num_pairs=args.num_pairs,
        seed=args.seed,
    )

    print(f"Generating {config.num_pairs} page/doodle pairs to {config.output_dir}...")
    print()

    stats = SynthPipeline(config).run()

    print()
    print(f"Done! Generated {stats.pages} pairs")
    print(f"Output: {stats.output_dir}")
    return 0


def cmd_synth_index(args: argparse.Namespace) -> int:
    from doodle_doc.synth.synth_index import SynthIndexer

    settings = get_settings()
    if args.config:
        settings = load_settings_from_yaml(Path(args.config))

    synth_dir = Path(args.synth_dir)
    if not synth_dir.exists():
        print(f"Error: Synthetic dataset not found: {synth_dir}", file=sys.stderr)
        return 1

    print(f"Indexing synthetic pages from: {synth_dir}/pages")
    print()

    indexer = SynthIndexer(settings, synth_dir)
    stats = indexer.run()

    print()
    print(f"Done! Indexed {stats.indexed} pages, skipped {stats.skipped}")
    return 0


def cmd_eval_synth(args: argparse.Namespace) -> int:
    from doodle_doc.eval.synth_eval import SynthEvalRunner

    settings = get_settings()
    if args.config:
        settings = load_settings_from_yaml(Path(args.config))

    synth_dir = Path(args.synth_dir)
    if not synth_dir.exists():
        print(f"Error: Synthetic dataset not found: {synth_dir}", file=sys.stderr)
        return 1

    print("\nEvaluating with ColQwen2...")

    runner = SynthEvalRunner(settings, synth_dir)
    result = runner.run(top_k=args.top_k)

    print()
    print(result.summary())

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
    index_parser.add_argument("--force", "-f", action="store_true", help="Force reindex")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", help="Host to bind to")
    serve_parser.add_argument("--port", "-p", type=int, help="Port to bind to")
    serve_parser.add_argument("--reload", "-r", action="store_true", help="Enable hot reload")

    eval_parser = subparsers.add_parser("eval", help="Run crop-based evaluation")
    eval_parser.add_argument("--config", "-c", help="Path to config YAML file")
    eval_parser.add_argument("--mode", choices=["fast", "accurate", "both"], default="both")
    eval_parser.add_argument("--regenerate", action="store_true", help="Regenerate pseudo-queries")
    eval_parser.add_argument("--num-queries", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--save-baseline", action="store_true", help="Save results as baseline")
    eval_parser.add_argument("--check-regression", action="store_true", help="Check against baseline")
    eval_parser.add_argument("--regression-threshold", type=float, default=0.05)

    synth_gen_parser = subparsers.add_parser("synth-generate", help="Generate synthetic dataset")
    synth_gen_parser.add_argument("--output", "-o", default="data/synth", help="Output directory")
    synth_gen_parser.add_argument("--num-pairs", "-n", type=int, default=25, help="Number of pairs")
    synth_gen_parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    synth_index_parser = subparsers.add_parser("synth-index", help="Index synthetic pages")
    synth_index_parser.add_argument("synth_dir", help="Path to synthetic dataset")
    synth_index_parser.add_argument("--config", "-c", help="Path to config YAML file")

    eval_synth_parser = subparsers.add_parser("eval-synth", help="Evaluate on synthetic dataset")
    eval_synth_parser.add_argument("synth_dir", help="Path to synthetic dataset")
    eval_synth_parser.add_argument("--config", "-c", help="Path to config YAML file")
    eval_synth_parser.add_argument("--top-k", "-k", type=int, default=20)

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "eval":
        return cmd_eval(args)
    elif args.command == "synth-generate":
        return cmd_synth_generate(args)
    elif args.command == "synth-index":
        return cmd_synth_index(args)
    elif args.command == "eval-synth":
        return cmd_eval_synth(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
