.PHONY: lint test verify fmt clippy help

# Default target
help:
	@echo "Available targets:"
	@echo "  make lint    - Run formatting and linting checks (fmt + clippy)"
	@echo "  make test    - Run local tests (library tests only)"
	@echo "  make verify  - Run all tests including Python comparison"
	@echo "  make fmt     - Check code formatting"
	@echo "  make clippy  - Run clippy linter"

# Check formatting without modifying files
fmt:
	@echo "Checking code formatting..."
	@cargo fmt --all -- --check

# Run clippy with all warnings as errors
clippy:
	@echo "Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

# Lint target: runs both fmt and clippy
lint: fmt clippy
	@echo "✓ All linting checks passed!"

# Run local tests (library tests only, no Python comparison)
test:
	@echo "Running library tests..."
	@cargo test --lib

# Verify target: runs local tests + Python comparison test
verify: test
	@echo "Running Python comparison tests..."
	@cargo test test_compare_with_python
	@echo "✓ All tests passed!"
