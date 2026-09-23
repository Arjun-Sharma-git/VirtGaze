.PHONY: install install-dev lint format typecheck check test test-cov clean run-tracker run-calibration export-model precommit precommit-install

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# AMD GPU (ROCm) — installs PyTorch ROCm wheel + onnxruntime-rocm
# Adjust --extra-index-url version to match your ROCm installation.
install-rocm:
	pip install -r requirements-rocm.txt
	pip install -e . --no-deps

# NVIDIA GPU (CUDA) — standard onnxruntime-gpu + existing torch CUDA wheel
install-cuda:
	pip install "onnxruntime-gpu>=1.15.0"
	pip install -e .

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

typecheck:
	mypy src/gaze_estimation

# Everything CI enforces, in one command
check: lint typecheck
	pytest tests/ -q

# ─── Git hooks ────────────────────────────────────────────────────────────────
precommit-install:
	pre-commit install

precommit:
	pre-commit run --all-files

# ─── Testing ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=gaze_estimation --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/ -v -k "not integration and not benchmark"

test-integration:
	pytest tests/ -v -k "integration"

# ─── Entry Points ─────────────────────────────────────────────────────────────
run-tracker:
	python scripts/run_tracker.py

run-calibration:
	python scripts/run_calibration.py

run-calibration-quick:
	python scripts/run_calibration.py --quick

benchmark-accuracy:
	python scripts/benchmark_accuracy.py

benchmark-latency:
	python scripts/benchmark_latency.py

calibrate-camera:
	python scripts/calibrate_camera.py

export-model:
	python scripts/export_model.py

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	rm -rf dist build

clean-profiles:
	find profiles/ -type f ! -name '.gitkeep' -delete

clean-models:
	find models/ -type f ! -name '.gitkeep' -delete
