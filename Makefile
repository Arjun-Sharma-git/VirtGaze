.PHONY: setup setup-dev setup-cpu setup-headless setup-rocm doctor doctor-camera install install-dev lint format typecheck check test test-cov clean run-tracker run-calibration export-model precommit precommit-install

# ─── Setup ────────────────────────────────────────────────────────────────────
# Prefer `make setup`: it creates .venv, picks the right OpenCV/PyTorch, installs
# the package and verifies the result.  The bare `install-*` targets below assume
# you already have the virtualenv you want to use activated.

setup:
	python scripts/bootstrap.py

setup-dev:
	python scripts/bootstrap.py --dev

setup-cpu:
	python scripts/bootstrap.py --torch cpu

setup-headless:
	python scripts/bootstrap.py --opencv headless

# AMD GPU (ROCm) — PyTorch ROCm wheel + onnxruntime-rocm
setup-rocm:
	python scripts/bootstrap.py --torch rocm

# Report versions, device, screen and camera without changing anything
doctor:
	python scripts/bootstrap.py --check

doctor-camera:
	python scripts/bootstrap.py --check --probe-camera

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
