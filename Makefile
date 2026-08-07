.PHONY: lint test verify fmt clippy parity-coverage setup-parity parity-version clean-parity help

# The crate version IS the pythermalcomfort version this port targets. Deriving the pin
# from Cargo.toml means CI and local runs can never drift from what is being ported,
# which is what previously let CI validate 3.9.x work against a 3.8.0 reference.
PTC_VERSION := $(shell grep -m1 '^version = ' Cargo.toml | cut -d'"' -f2)

# Virtualenv holding the reference pythermalcomfort. Kept inside the repo (gitignored)
# rather than /tmp so it survives reboots. Override with PARITY_VENV=... if needed.
PARITY_VENV ?= .parity-venv
PYTHON ?= python3

# Resolved lazily (=, not :=) because the venv may not exist when the Makefile is parsed.
# If there is no venv we fall through to whatever python is ambient — which is correct for
# CI, where pythermalcomfort is installed into the runner's system python. Either way the
# test_pythermalcomfort_version_matches_crate guard fails the run if the version is wrong,
# so this only decides *where* the reference comes from, never *whether* it is checked.
PARITY_SITE_PACKAGES = $(shell ls -d $(PARITY_VENV)/lib/python*/site-packages 2>/dev/null)
PARITY_ENV = $(if $(PARITY_SITE_PACKAGES),PYTHONPATH=$(PARITY_SITE_PACKAGES),)

# Default target
help:
	@echo "Available targets:"
	@echo "  make setup-parity   - Create the reference venv with pythermalcomfort==$(PTC_VERSION)"
	@echo "  make lint           - Run fmt + clippy + parity coverage check"
	@echo "  make test           - Run the full suite in both no_std and std configurations"
	@echo "  make verify         - Run lint + the full suite including Python parity tests"
	@echo "  make parity-coverage- Check every public model/utility has a parity test"
	@echo "  make parity-version - Print the pythermalcomfort version this port targets"
	@echo "  make fmt            - Check code formatting"
	@echo "  make clippy         - Run clippy linter"
	@echo "  make clean-parity   - Remove the reference venv"

parity-version:
	@echo $(PTC_VERSION)

# Build the reference environment. Idempotent: re-running repins to the current version,
# so this is also how you move the reference forward during a version bump.
setup-parity:
	@echo "Creating parity venv at $(PARITY_VENV) with pythermalcomfort==$(PTC_VERSION)..."
	@$(PYTHON) -m venv $(PARITY_VENV)
	@$(PARITY_VENV)/bin/pip install --quiet --upgrade pip
	@$(PARITY_VENV)/bin/pip install --quiet pythermalcomfort==$(PTC_VERSION)
	@echo "✓ Reference pythermalcomfort $$($(PARITY_VENV)/bin/python -c 'import pythermalcomfort; print(pythermalcomfort.__version__)') ready"

clean-parity:
	@rm -rf $(PARITY_VENV)
	@echo "✓ Removed $(PARITY_VENV)"

# Check formatting without modifying files
fmt:
	@echo "Checking code formatting..."
	@cargo fmt --all -- --check

# Run clippy with all warnings as errors
clippy:
	@echo "Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

# Every public model/utility must have a cross-library parity test
parity-coverage:
	@echo "Checking parity test coverage..."
	@$(PYTHON) scripts/check_parity_coverage.py

# Lint target: formatting, clippy, and parity coverage
lint: fmt clippy parity-coverage
	@echo "✓ All linting checks passed!"

# Run the whole suite in BOTH supported configurations. The crate ships no_std by
# default with an optional std accuracy path, so a green run in one proves nothing
# about the other. This includes unit, integration and doc tests.
test:
	@if [ -z "$(PARITY_SITE_PACKAGES)" ]; then \
		echo "No parity venv at $(PARITY_VENV); using ambient python."; \
		echo "Run 'make setup-parity' if the version guard fails."; \
	fi
	@echo "Running full suite (no_std default) against pythermalcomfort $(PTC_VERSION)..."
	@$(PARITY_ENV) cargo test --release
	@echo "Running full suite (std feature) against pythermalcomfort $(PTC_VERSION)..."
	@$(PARITY_ENV) cargo test --release --features std
	@echo "✓ Both configurations passed!"

# Verify target: linting plus the complete test suite
verify: lint test
	@echo "✓ All checks passed!"
