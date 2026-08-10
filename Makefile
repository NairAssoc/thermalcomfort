.PHONY: lint test verify fmt clippy parity-coverage setup-parity parity-version clean-parity sweep no-std-check help

# The crate version IS the pythermalcomfort version this port targets. Deriving the pin
# from Cargo.toml means CI and local runs can never drift from what is being ported,
# which is what previously let CI validate 3.9.x work against a 3.8.0 reference.
#
# Parsed via cargo, not grep: `grep -m1 '^version = '` picked up whichever table came
# first, so a [dependencies] block above [package] silently yielded a dependency's
# version. A trailing `-rc.1` marks a revision of the port, not of upstream, so it is
# stripped - PEP 440 spells pre-releases differently and no such upstream sdist exists.
PTC_VERSION := $(firstword $(subst -, ,$(shell cargo metadata --no-deps --format-version 1 \
	| sed -n 's/.*"name":"thermalcomfort","version":"\([^"]*\)".*/\1/p')))

# Virtualenv holding the reference pythermalcomfort. Kept inside the repo (gitignored)
# rather than /tmp so it survives reboots. Override with PARITY_VENV=... if needed.
PARITY_VENV ?= .parity-venv
PYTHON ?= python3

# Resolved lazily (=, not :=) because the venv may not exist when the Makefile is parsed.
# If there is no venv we fall through to whatever python is ambient — which is correct for
# CI, where pythermalcomfort is installed into the runner's system python. Either way the
# test_pythermalcomfort_version_matches_crate guard fails the run if the version is wrong,
# so this only decides *where* the reference comes from, never *whether* it is checked.
#
# Asks the venv's own interpreter rather than globbing `lib/python*/site-packages`:
# after a Python minor upgrade the glob matches several directories, and make then
# tries to execute the second one as a command.
PARITY_SITE_PACKAGES = $(shell $(PARITY_VENV)/bin/python -c \
	'import site; print(site.getsitepackages()[0])' 2>/dev/null)
PARITY_ENV = $(if $(PARITY_SITE_PACKAGES),PYTHONPATH=$(PARITY_SITE_PACKAGES),)

ifeq ($(strip $(PTC_VERSION)),)
$(error Could not determine the crate version from cargo metadata. The parity \
        reference version is derived from it, so refusing to continue.)
endif

# `make -j2 verify` would otherwise interleave lint and test, defeating the intended
# ordering and contending on the cargo target-dir lock.
.NOTPARALLEL:

# Default target
help:
	@echo "Available targets:"
	@echo "  make setup-parity   - Create the reference venv with pythermalcomfort==$(PTC_VERSION)"
	@echo "  make lint           - Run fmt + clippy + parity coverage check"
	@echo "  make test           - Run the full suite in both no_std and std configurations"
	@echo "  make sweep          - Deep randomised differential sweep (SWEEP_N=$(SWEEP_N))"
	@echo "  make no-std-check   - Build for $(NO_STD_TARGET) to prove no_std still holds"
	@echo "  make verify         - Run lint + the full suite including Python parity tests"
	@echo "  make parity-coverage- Check parity tests exist AND every upstream name is ported"
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

# Every public model/utility must have a cross-library parity test, and every
# pythermalcomfort name must have a Rust port. The second direction needs the reference
# package importable, hence PARITY_ENV: it inventories the *installed* API, because only
# that can reveal something upstream has and this port does not.
parity-coverage:
	@echo "Checking parity test coverage..."
	@$(PARITY_ENV) PTC_VERSION=$(PTC_VERSION) $(PYTHON) scripts/check_parity_coverage.py

# Lint target: formatting, clippy, and parity coverage
lint: fmt clippy parity-coverage
	@echo "✓ All linting checks passed!"

# The crate has one configuration. It used to have two - a no_std default and a std
# accuracy path - and both had to run because a green result in one proved nothing about
# the other. The std feature is now a no-op, so a single run covers everything. The
# wasm32 build is still checked separately: `cargo test` links the std test harness, so
# it cannot prove no_std compiles.
test:
	@if [ -z "$(PARITY_SITE_PACKAGES)" ]; then \
		echo "No parity venv at $(PARITY_VENV); using ambient python."; \
		echo "Run 'make setup-parity' if the version guard fails."; \
	fi
	@echo "Running full suite against pythermalcomfort $(PTC_VERSION)..."
	@$(PARITY_ENV) cargo test --release
	@echo "✓ Suite passed!"

# `cargo test` links the std test harness, so a green suite says nothing about whether the
# crate still compiles without std. This builds for a genuine no_std target, which is the
# only thing that does. CI installs the target; locally it is skipped with a notice rather
# than failing, because a missing rustup target is a setup gap, not a code defect.
NO_STD_TARGET ?= wasm32-unknown-unknown
no-std-check:
	@if rustup target list --installed 2>/dev/null | grep -q '^$(NO_STD_TARGET)$$'; then \
		echo "Building for $(NO_STD_TARGET) to prove no_std still holds..."; \
		cargo build --quiet --target $(NO_STD_TARGET); \
		echo "✓ no_std target builds!"; \
	else \
		echo "SKIPPED: $(NO_STD_TARGET) not installed, so no_std was NOT verified."; \
		echo "  Install it with: rustup target add $(NO_STD_TARGET)"; \
	fi

# Deep randomised differential sweep. `make test` already runs a short one as part of
# the suite; this is the long-form version for a release check or a bug hunt. Failures
# print the seed, so a divergence found here is reproducible at this N.
SWEEP_N ?= 20000
sweep:
	@echo "Running differential sweep with SWEEP_N=$(SWEEP_N)..."
	@$(PARITY_ENV) SWEEP_N=$(SWEEP_N) cargo test --release \
		--test differential_sweep -- --nocapture

# Verify target: linting plus the complete test suite
verify: lint test no-std-check
	@echo "✓ All checks passed!"
