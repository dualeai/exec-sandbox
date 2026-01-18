# Misc
name ?= exec_sandbox
python_version ?= 3.12  # Lowest compatible version (see pyproject.toml requires-python)

# Versions
version_full ?= $(shell $(MAKE) --silent version-full)
version_small ?= $(shell $(MAKE) --silent version)

version:
	@bash ./cicd/version.sh -g . -c

version-full:
	@bash ./cicd/version.sh -g . -c -m

version-pypi:
	@bash ./cicd/version.sh -g .

# ============================================================================
# Installation
# ============================================================================

install-sys:
	brew install qemu

install:
	uv venv --python $(python_version) --allow-existing
	$(MAKE) install-deps
	$(MAKE) --directory guest-agent install
	$(MAKE) --directory tiny-init install
	$(MAKE) --directory gvproxy-wrapper install

install-deps:
	uv sync --extra dev --extra s3

# ============================================================================
# Upgrade (auto-called targets for dependency updates)
# ============================================================================

upgrade:
	uv lock --upgrade --refresh
	$(MAKE) build-catalogs
	$(MAKE) --directory guest-agent upgrade
	$(MAKE) --directory tiny-init upgrade
	$(MAKE) --directory gvproxy-wrapper upgrade

# Build package allow-lists from PyPI and npm registries
build-catalogs:
	@echo "📦 Building package catalogs (PyPI + npm top 10k)..."
	uv run --script scripts/build_package_catalogs.py src/exec_sandbox/resources

# ============================================================================
# Image Building
# ============================================================================

build-images:
	@echo "🔨 Building QEMU base images..."
	./scripts/build-images.sh

# ============================================================================
# Testing
# ============================================================================

test:
	$(MAKE) test-static
	$(MAKE) test-func

test-static:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright .
	uv run -m vulture src/ scripts/ --min-confidence 80
	$(MAKE) --directory guest-agent test-static
	$(MAKE) --directory tiny-init test-static
	$(MAKE) --directory gvproxy-wrapper test-static

# All tests together for accurate coverage measurement (excludes sudo tests)
test-func:
	uv run pytest tests/ -v -n auto -m "not sudo"

# Tests requiring sudo privileges (run separately with elevated permissions)
test-sudo:
	uv run pytest tests/ -v -n auto -m "sudo"

# Unit tests only (fast)
test-unit:
	uv run pytest tests/ -v -n auto -m "unit"
	$(MAKE) --directory guest-agent test-unit
	$(MAKE) --directory tiny-init test-unit
	$(MAKE) --directory gvproxy-wrapper test-unit

# ============================================================================
# Linting
# ============================================================================

lint:
	uv run ruff format .
	uv run ruff check --fix .
	$(MAKE) --directory guest-agent lint
	$(MAKE) --directory tiny-init lint
	$(MAKE) --directory gvproxy-wrapper lint

# ============================================================================
# CI Monitoring (requires: gh cli)
# Usage: make ci-status [run_id=ID] or make ci-diagnose [run_id=ID]
# ============================================================================

run_id ?=

ci-status:
	@./scripts/ci-diagnose.sh status $(run_id)

ci-diagnose:
	@./scripts/ci-diagnose.sh diagnose $(run_id)

# ============================================================================
# Benchmarking (concurrent VM latency)
# ============================================================================

bench:
	uv run python scripts/benchmark_latency.py -n 10

bench-pool:
	uv run python scripts/benchmark_latency.py -n 10 --pool 8

# ============================================================================
# Building
# ============================================================================

build:
	$(MAKE) --directory guest-agent build
	$(MAKE) --directory tiny-init build
	$(MAKE) --directory gvproxy-wrapper build

# ============================================================================
# Cleanup
# ============================================================================

clean:
	$(MAKE) --directory guest-agent clean
	$(MAKE) --directory tiny-init clean
	$(MAKE) --directory gvproxy-wrapper clean
	rm -rf .pytest_cache .coverage htmlcov
