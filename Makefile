.PHONY: lint test verify fmt clippy doc examples check parity-coverage setup-parity setup-toolchain parity-version clean-parity sweep sweep-ci no-std-check beta help

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

# Every cargo invocation goes through $(CARGO) so a whole target can be re-run on another
# toolchain: `make lint test TOOLCHAIN=beta` is what CI's beta matrix leg does.
TOOLCHAIN ?=
CARGO = cargo$(if $(TOOLCHAIN), +$(TOOLCHAIN),)

# The no_std proof needs two targets: a bare-metal one, where there is no std to fall
# back on at all, and wasm32, which is also built in release because that is how it ships.
NO_STD_TARGETS = thumbv7em-none-eabihf wasm32-unknown-unknown

# `make sweep` is the deep run; this is the depth CI runs on every push, and the depth
# `make verify` runs so that a push cannot fail CI's sweep without first failing here.
CI_SWEEP_N ?= 2000

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
	@echo "  make setup-parity    - Create the reference venv with pythermalcomfort==$(PTC_VERSION)"
	@echo "  make setup-toolchain - Install the rustup targets and beta toolchain verify needs"
	@echo "  make verify          - Everything CI runs: lint + check + suite + no_std + sweep + beta"
	@echo "  make lint            - fmt + clippy + rustdoc + examples + cargo check + parity coverage"
	@echo "  make test            - Run the full suite against the reference pythermalcomfort"
	@echo "  make no-std-check    - Check $(NO_STD_TARGETS); build wasm32 in release"
	@echo "  make sweep-ci        - Differential sweep at CI's depth (SWEEP_N=$(CI_SWEEP_N))"
	@echo "  make sweep           - Deep randomised differential sweep (SWEEP_N=$(SWEEP_N))"
	@echo "  make beta            - lint + test on the beta toolchain, as CI's matrix does"
	@echo "  make parity-coverage - Check parity tests exist AND every upstream name is ported"
	@echo "  make parity-version  - Print the pythermalcomfort version this port targets"
	@echo "  make fmt             - Check code formatting"
	@echo "  make clippy          - Run clippy linter"
	@echo "  make doc             - Build the docs with rustdoc warnings as errors"
	@echo "  make examples        - Build and run the examples"
	@echo "  make check           - cargo check --all-targets"
	@echo "  make clean-parity    - Remove the reference venv"
	@echo ""
	@echo "Any target takes TOOLCHAIN=<name>, e.g. make test TOOLCHAIN=beta."

parity-version:
	@echo $(PTC_VERSION)

# Build the reference environment. Idempotent: re-running repins to the current version,
# so this is also how you move the reference forward during a version bump. Dependencies
# are upgraded eagerly so a re-run resolves what a fresh CI runner resolves today, rather
# than keeping whatever numba and scipy the venv was first created with.
setup-parity:
	@echo "Creating parity venv at $(PARITY_VENV) with pythermalcomfort==$(PTC_VERSION)..."
	@$(PYTHON) -m venv $(PARITY_VENV)
	@$(PARITY_VENV)/bin/pip install --quiet --upgrade pip
	@$(PARITY_VENV)/bin/pip install --quiet --upgrade --upgrade-strategy eager \
		pythermalcomfort==$(PTC_VERSION)
	@echo "✓ Reference pythermalcomfort $$($(PARITY_VENV)/bin/python -c 'import pythermalcomfort; print(pythermalcomfort.__version__)') ready"

clean-parity:
	@rm -rf $(PARITY_VENV)
	@echo "✓ Removed $(PARITY_VENV)"

# Everything `make verify` needs beyond a stable toolchain. Idempotent.
setup-toolchain:
	@rustup target add $(NO_STD_TARGETS)
	@rustup toolchain install beta --component rustfmt,clippy
	@echo "✓ Toolchain ready for make verify"

# Check formatting without modifying files
fmt:
	@echo "Checking code formatting..."
	@$(CARGO) fmt --all -- --check

# Run clippy with all warnings as errors
clippy:
	@echo "Running clippy..."
	@$(CARGO) clippy --all-targets --all-features -- -D warnings

# CI builds the docs with warnings denied, so a public doc comment linking to a private
# item fails there; run the same check here so `make verify` catches it first.
doc:
	@echo "Building docs with rustdoc warnings as errors..."
	@RUSTDOCFLAGS="-D warnings" $(CARGO) doc --no-deps --quiet

# CI builds every example and runs the two documented ones.
examples:
	@echo "Building and running examples..."
	@$(CARGO) build --examples --quiet
	@$(CARGO) run --quiet --example basic_pmv > /dev/null
	@$(CARGO) run --quiet --example typed_api > /dev/null

check:
	@echo "Checking every target compiles..."
	@$(CARGO) check --all-targets --quiet

# Every public model/utility must have a cross-library parity test, and every
# pythermalcomfort name must have a Rust port. The second direction needs the reference
# package importable, hence PARITY_ENV: it inventories the *installed* API, because only
# that can reveal something upstream has and this port does not.
#
# PYTHONIOENCODING is forced to cp1252 because that is the console CI's Windows runners
# give Python: a script that prints anything non-ASCII without setting its own output
# encoding fails there, and should fail here first.
parity-coverage:
	@echo "Checking parity test coverage..."
	@$(PARITY_ENV) PTC_VERSION=$(PTC_VERSION) PYTHONIOENCODING=cp1252:strict \
		$(PYTHON) scripts/check_parity_coverage.py

# Lint target: everything CI's per-commit jobs check, in the same form
lint: fmt clippy doc examples check parity-coverage
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
	@$(PARITY_ENV) $(CARGO) test --release
	@echo "✓ Suite passed!"

# `cargo test` links the std test harness, so a green suite says nothing about whether the
# crate still compiles without std. Only a build for a genuine no_std target does. A missing
# rustup target fails the check rather than skipping it: a skip once let `make verify` pass
# locally on a tree CI then rejected, which is the one thing verify exists to prevent.
no-std-check:
	@for target in $(NO_STD_TARGETS); do \
		if ! rustup target list --installed --toolchain $(if $(TOOLCHAIN),$(TOOLCHAIN),stable) 2>/dev/null | grep -q "^$$target$$"; then \
			echo "✗ $$target is not installed, so no_std cannot be verified. Run: make setup-toolchain"; \
			exit 1; \
		fi; \
	done
	@echo "Checking $(NO_STD_TARGETS) to prove no_std still holds..."
	@for target in $(NO_STD_TARGETS); do $(CARGO) check --quiet --target $$target || exit 1; done
	@$(CARGO) build --quiet --release --target wasm32-unknown-unknown
	@echo "✓ no_std targets build!"

# Deep randomised differential sweep. `make test` already runs a short one as part of
# the suite; this is the long-form version for a release check or a bug hunt. Failures
# print the seed, so a divergence found here is reproducible at this N.
SWEEP_N ?= 20000
sweep:
	@echo "Running differential sweep with SWEEP_N=$(SWEEP_N)..."
	@$(PARITY_ENV) SWEEP_N=$(SWEEP_N) $(CARGO) test --release \
		--test differential_sweep -- --nocapture

sweep-ci:
	@$(MAKE) --no-print-directory sweep SWEEP_N=$(CI_SWEEP_N)

# CI's test matrix runs lint and the suite on beta as well as stable, and beta is where
# new clippy and rustdoc lints land first. Fails, not skips, when beta is absent.
beta:
	@if ! rustup toolchain list 2>/dev/null | grep -q '^beta-'; then \
		echo "✗ The beta toolchain is not installed, so CI's beta leg cannot be reproduced. Run: make setup-toolchain"; \
		exit 1; \
	fi
	@echo "Repeating lint and the suite on the beta toolchain..."
	@$(MAKE) --no-print-directory lint test TOOLCHAIN=beta

# The union of every CI job, so a tree that passes here cannot fail there for a reason
# the tree controls. CI invokes these same targets; add a check to one of them and both
# sides get it. Ordered fastest-failing first.
verify: lint check test no-std-check beta sweep-ci
	@echo "✓ All checks passed!"
