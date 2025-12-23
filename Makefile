.PHONY: dev index serve test lint eval eval-fast eval-accurate eval-baseline eval-check

# Development
dev:
	uv run uvicorn doodle_doc.api.main:app --reload --port 8000

# Run indexing
index:
	uv run doodle-doc index $(ROOT)

# Run full server (API + UI)
serve:
	uv run uvicorn doodle_doc.api.main:app --host 0.0.0.0 --port 8000

# Testing
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src/doodle_doc --cov-report=html

# Linting
lint:
	uv run ruff check src/ tests/
	uv run mypy src/

# Evaluation
eval:
	uv run doodle-doc eval --config configs/default.yaml

eval-fast:
	uv run doodle-doc eval --config configs/default.yaml --mode fast

eval-accurate:
	uv run doodle-doc eval --config configs/default.yaml --mode accurate

eval-baseline:
	uv run doodle-doc eval --config configs/default.yaml --save-baseline

eval-check:
	uv run doodle-doc eval --config configs/default.yaml --check-regression

# UI (run from ui/ directory)
ui-dev:
	cd ui && npm run dev

ui-build:
	cd ui && npm run build