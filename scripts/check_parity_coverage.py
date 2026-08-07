#!/usr/bin/env python3
"""Fail if any public model/utility function lacks a cross-library parity test.

The port's correctness claim rests entirely on comparing against pythermalcomfort. A
function with no parity test is unverified no matter how many unit tests it has, because
unit tests assert against values a human transcribed once — they cannot notice upstream
changing. This check makes that gap a build failure rather than a matter of discipline.

Run via `make parity-coverage` (included in `make lint`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
PARITY_TESTS = REPO / "tests" / "python_comparison.rs"

# Functions with NO pythermalcomfort counterpart, so there is nothing to compare against.
# Each entry needs a reason. "Not done yet" is not a reason — that is KNOWN_GAPS.
EXEMPT: dict[str, str] = {
    "humidex_masterson": "variant not present in pythermalcomfort",
    "brentq": "numerical root-finder; Python delegates to scipy",
    "round_to": "formatting helper, no Python equivalent",
    "valid_range": "internal applicability helper, exercised via every model that gates on it",
    "celsius_to_temp": "Rust measurement-type adapter, no Python analogue",
    "temp_to_celsius": "Rust measurement-type adapter, no Python analogue",
    "ms_to_speed": "Rust measurement-type adapter, no Python analogue",
    "speed_to_ms": "Rust measurement-type adapter, no Python analogue",
    "pmv_ppd_iso_typed": "typed wrapper over pmv_ppd_iso, which is covered",
    "pmv_ppd_ashrae_typed": "typed wrapper over pmv_ppd_ashrae, which is covered",
    "body_surface_area_dubois": "Rust splits this out; Python's body_surface_area takes a formula argument",
}

# Functions that DO have a pythermalcomfort counterpart but no parity test yet. This is a
# backlog, not an exemption: it exists so that pre-existing gaps do not block the check
# from catching NEW untested functions. Entries should only ever be removed.
#
# Recorded 2026-08-07 while adding this check. Every one of these is testable today.
KNOWN_GAPS: dict[str, str] = {
    "use_fans_heatwaves": "pythermalcomfort.models.use_fans_heatwaves",
    "body_surface_area": "pythermalcomfort.utilities.body_surface_area",
    "clo_area_factor": "pythermalcomfort.utilities.clo_area_factor",
    "clo_correction_factor_environment": "pythermalcomfort.utilities.clo_correction_factor_environment",
    "clo_dynamic_ashrae": "pythermalcomfort.utilities.clo_dynamic_ashrae",
    "clo_insulation_air_layer": "pythermalcomfort.utilities.clo_insulation_air_layer",
    "clo_total_insulation": "pythermalcomfort.utilities.clo_total_insulation",
    "enthalpy_air": "pythermalcomfort.utilities.enthalpy_air",
    "mean_radiant_temperature": "pythermalcomfort.utilities.mean_radiant_tmp",
    "operative_temperature": "pythermalcomfort.utilities.operative_tmp",
    "p_sat_antoine": "pythermalcomfort.utilities.antoine",
    "running_mean_outdoor_temperature": "pythermalcomfort.utilities.running_mean_outdoor_temperature",
    "f_svv": "pythermalcomfort.utilities.f_svv",
    "transpose_sharp_altitude": "pythermalcomfort.utilities.transpose_sharp_altitude",
}


def public_function_names() -> set[str]:
    """Collect the crate's publicly callable snake_case functions.

    Two sources, because neither alone is complete:
      * `pub use` re-exports in src/**/mod.rs and src/lib.rs — the curated API surface.
      * module-level `pub fn` in src/*.rs and src/models/*.rs — functions reachable as
        `thermalcomfort::utilities::foo` that are never re-exported at the root.

    Only column-0 `pub fn` counts: rustfmt indents methods inside `impl` blocks, and
    `cargo fmt --check` is enforced, so indentation reliably distinguishes the two.
    """
    names: set[str] = set()

    rust_files = sorted(SRC.rglob("*.rs"))

    for path in rust_files:
        text = path.read_text()

        # Module-level free functions
        names.update(re.findall(r"^pub fn (\w+)", text, re.MULTILINE))

        # Re-exports. Take only the items being imported, not the path segments, so a
        # function sharing its name with its module (e.g. `ireq`) is not lost.
        for block in re.findall(r"^pub use ([^;]+);", text, re.MULTILINE):
            braced = re.search(r"\{(.*)\}", block, re.DOTALL)
            if braced:
                items = braced.group(1).split(",")
            else:
                items = [block.rsplit("::", 1)[-1]]
            for item in items:
                item = item.strip().split(" as ")[0].strip()
                if re.fullmatch(r"[a-z_][a-z0-9_]*", item):
                    names.add(item)

    # Constants are SCREAMING_CASE and types are CamelCase, so both are already excluded
    # by the snake_case filter. Drop anything that is only a module name.
    module_names = set()
    for path in rust_files:
        module_names.update(re.findall(r"^pub mod (\w+);", path.read_text(), re.MULTILINE))
    # A name that is *also* a real function stays; only pure modules are dropped.
    module_only = {
        n for n in names if n in module_names and not _is_defined_fn(n, rust_files)
    }
    return names - module_only


def _is_defined_fn(name: str, rust_files: list[Path]) -> bool:
    """True if `name` is defined as a module-level `pub fn` anywhere in the crate."""
    pattern = re.compile(rf"^pub fn {re.escape(name)}\b", re.MULTILINE)
    return any(pattern.search(p.read_text()) for p in rust_files)


def main() -> int:
    if not PARITY_TESTS.exists():
        print(f"error: {PARITY_TESTS} not found", file=sys.stderr)
        return 1

    # Strip `use ...;` statements first: a name appearing only in an import is not
    # evidence of a parity test, and counting it would hide exactly the gap we look for.
    tests = re.sub(r"^use [^;]+;", "", PARITY_TESTS.read_text(), flags=re.MULTILINE)
    functions = public_function_names()

    def is_tested(name: str) -> bool:
        return bool(re.search(rf"\b{re.escape(name)}\b", tests))

    untested = {n for n in functions if n not in EXEMPT and not is_tested(n)}

    # New, unaccounted-for gaps. These fail the build.
    missing = sorted(untested - KNOWN_GAPS.keys())
    # Backlog entries still outstanding. Reported, but do not fail.
    outstanding = sorted(untested & KNOWN_GAPS.keys())

    failed = False

    # A list entry that is no longer accurate is rot; surface it so the lists stay honest.
    for label, listing in (("EXEMPT", EXEMPT), ("KNOWN_GAPS", KNOWN_GAPS)):
        stale = sorted(n for n in listing if n not in functions)
        if stale:
            failed = True
            print(
                f"Stale entries in {label} (no longer a public function, remove them):",
                file=sys.stderr,
            )
            for name in stale:
                print(f"  - {name}", file=sys.stderr)

    resolved = sorted(KNOWN_GAPS.keys() & functions & {n for n in functions if is_tested(n)})
    if resolved:
        failed = True
        print(
            f"\n{len(resolved)} function(s) now have parity tests but are still listed "
            f"in KNOWN_GAPS. Remove them so the backlog stays accurate:\n",
            file=sys.stderr,
        )
        for name in resolved:
            print(f"  - {name}", file=sys.stderr)

    if missing:
        failed = True
        print(
            f"\n{len(missing)} public function(s) have no parity test in "
            f"{PARITY_TESTS.relative_to(REPO)}:\n",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nAdd a test that calls the pythermalcomfort function through pyo3 and "
            "compares against the Rust output.\n"
            "If there is genuinely no Python counterpart, add it to EXEMPT with a reason.\n",
            file=sys.stderr,
        )

    if failed:
        return 1

    tested = len(functions) - len(EXEMPT) - len(outstanding)
    print(f"✓ {tested} public functions have parity tests, {len(EXEMPT)} exempt")
    if outstanding:
        print(f"  {len(outstanding)} known gap(s) outstanding: {', '.join(outstanding)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
