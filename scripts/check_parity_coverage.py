#!/usr/bin/env python3
"""Check parity coverage in both directions.

1. Rust -> Python: every public Rust function has a cross-library parity test. A function
   with no parity test is unverified no matter how many unit tests it has, because unit
   tests assert against values a human transcribed once — they cannot notice upstream
   changing.

2. Python -> Rust: every public pythermalcomfort name has a Rust port. This direction was
   missing until 2026-08-09, and its absence is not hypothetical: `JOS3`, an entire
   17-segment thermoregulation model, was absent from the port while the build stayed
   green and the README claimed "100% Feature Complete". Direction 1 cannot see that, by
   construction — it only ever asks questions about names the port already has.

Direction 2 imports the installed pythermalcomfort, so `make parity-coverage` now needs
the reference package importable and refuses to run against the wrong version. It fails
loudly rather than skipping: a completeness check that silently does nothing when the
reference is missing is the same hole it was written to close.

Run via `make parity-coverage` (included in `make lint`).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
# Both targets count as parity coverage: the hand-written cases pin specific known bugs,
# the randomised sweep covers the space. A function verified by either is verified.
PARITY_TEST_FILES = [
    REPO / "tests" / "python_comparison.rs",
    REPO / "tests" / "differential_sweep.rs",
]

# Matches a module-level public function, including the qualifiers Rust allows between
# `pub` and `fn`. Anchored at column 0 so `impl` methods (which rustfmt indents, and
# `cargo fmt --check` enforces) are excluded.
PUB_FN_RE = r'^pub (?:(?:const|unsafe|async|extern\s+"[^"]*")\s+)*fn (\w+)'

# Functions with NO pythermalcomfort counterpart, so there is nothing to compare against.
# Each entry needs a reason. "Not done yet" is not a reason — that is KNOWN_GAPS.
EXEMPT: dict[str, str] = {
    "brentq": "numerical root-finder; Python delegates to scipy",
    "round_to": "formatting helper, no Python equivalent",
    "round_half_even": "mirrors numpy.around's tie rule; numpy, not pythermalcomfort",
    "valid_range": "internal applicability helper, exercised via every model that gates on it",
    "celsius_to_temp": "Rust measurement-type adapter, no Python analogue",
    "temp_to_celsius": "Rust measurement-type adapter, no Python analogue",
    "ms_to_speed": "Rust measurement-type adapter, no Python analogue",
    "speed_to_ms": "Rust measurement-type adapter, no Python analogue",
    "pmv_ppd_iso_typed": "typed wrapper over pmv_ppd_iso, which is covered",
    "pmv_ppd_ashrae_typed": "typed wrapper over pmv_ppd_ashrae, which is covered",
    "body_surface_area_dubois": "Rust splits this out; Python's body_surface_area takes a formula argument",
}

# Functions that have a pythermalcomfort counterpart but no parity test yet. A backlog,
# not an exemption: it exists so pre-existing gaps do not stop the check catching NEW
# untested functions. Entries should only ever be removed.
#
# Emptied 2026-08-09 - every public function with a Python counterpart now has a parity
# test. Adding an entry here is a regression; prefer writing the test.
KNOWN_GAPS: dict[str, str] = {}


def executable_test_source(text: str) -> str:
    """Reduce the parity test file to code that actually runs.

    Removes `use` statements, comments, and `#[ignore]`d test bodies. Each has been
    observed to make an untested function look covered.

    String literals are deliberately NOT stripped: doing so requires pairing quotes,
    and one unbalanced quote makes the regex swallow the rest of the file (which
    silently marked all 50 functions untested when tried). The call-shaped match in
    `is_tested` already ignores prose, and a name inside a string is not call-shaped.
    """
    text = re.sub(r"^use [^;]+;", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    # An #[ignore]d test cannot verify anything, so drop those bodies too.
    out, i = [], 0
    for match in re.finditer(r"#\[ignore[^\]]*\]", text):
        brace = text.find("{", match.end())
        if brace == -1:
            continue
        depth, j = 0, brace
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(text[i:match.start()])
        i = j
    out.append(text[i:])
    return "".join(out)


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

        # Module-level free functions. The qualifier group matters: `pub const fn` is
        # already used in this crate, and a bare `^pub fn` regex would let the next
        # module-level one into the public API with no parity requirement at all.
        names.update(re.findall(PUB_FN_RE, text, re.MULTILINE))

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
    pattern = re.compile(
        PUB_FN_RE.replace(r"(\w+)", re.escape(name)) + r"\b", re.MULTILINE
    )
    return any(pattern.search(p.read_text()) for p in rust_files)


def public_rust_identifiers() -> set[str]:
    """Every publicly reachable Rust name: functions, structs, enums and type aliases.

    The Python direction needs more than functions, because pythermalcomfort exposes
    some of its API as classes (`JOS3`, `Sports`) that a Rust port would spell as a
    struct rather than a `fn`.
    """
    names = public_function_names()
    for path in sorted(SRC.rglob("*.rs")):
        text = path.read_text()
        names.update(
            re.findall(r"^pub (?:struct|enum|type|trait) (\w+)", text, re.MULTILINE)
        )
    return names


def python_public_api() -> dict[str, str]:
    """Map every public pythermalcomfort name to the module it came from.

    Imported rather than parsed: the point of this direction is to notice when
    *upstream* grows something the port does not have, and only the installed package
    knows that.
    """
    import pythermalcomfort.models as py_models
    import pythermalcomfort.utilities as py_utilities

    api: dict[str, str] = {}
    for label, module in (("models", py_models), ("utilities", py_utilities)):
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if not (callable(obj) or isinstance(obj, type)):
                continue
            # Only things pythermalcomfort itself defines; skip re-exported numpy etc.
            origin = getattr(obj, "__module__", "") or ""
            if not origin.startswith("pythermalcomfort"):
                continue
            api[name] = label
    return api


# pythermalcomfort names the Rust port spells differently.
PYTHON_TO_RUST: dict[str, str] = {
    "wet_bulb_tmp": "wet_bulb_temperature",
    "dew_point_tmp": "dew_point_temperature",
    "mean_radiant_tmp": "mean_radiant_temperature",
    "operative_tmp": "operative_temperature",
    "BodySurfaceAreaEquations": "BsaFormula",
    "Postures": "Posture",
    "JOS3": "Jos3Model",
}

# pythermalcomfort names that need no Rust counterpart. Each needs a reason.
PYTHON_EXEMPT: dict[str, str] = {
    "units_converter": "the Rust API takes typed quantities, so unit conversion is the type system's job",
    "valid_range": "internal applicability helper; the port applies the same masks inside each model",
    "validate_type": "Python runtime type checking; Rust does this at compile time",
    "adaptive_cooling_effect": "implemented as a private helper in src/models/adaptive.rs and exercised through both adaptive models",
    "DefaultSkinTemperature": "a per-body-part default-skin-temperature NamedTuple in "
    "utilities.py, referenced nowhere else in pythermalcomfort (confirmed by grepping the "
    "installed package) -- not even by JOS3, now ported, which seeds skin temperature from "
    "its own internal Default.skin_temperature = 34 (a single scalar, ported as "
    "defaults::SKIN_TEMPERATURE). Dead public data in the library being ported, not a used "
    "constant, so there is nothing for a Rust port to call.",
    "Models": "a single enum of every standard upstream supports; the port encodes the "
    "choice per function instead - Iso7933Model for PHS, use_iso/use_ashrae flags for the "
    "psychrometric helpers - so there is no one type to map it to",
}

# pythermalcomfort API with no Rust port yet. A backlog, not an exemption: reported on
# every run but does not fail, so that pre-existing gaps cannot mask a NEW one appearing
# upstream. Entries should only ever be removed.
PYTHON_NOT_PORTED: dict[str, str] = {}


def check_python_direction() -> tuple[bool, list[str]]:
    """Fail if pythermalcomfort exposes something the port has not implemented.

    This is the direction the Rust-side check cannot see. `make parity-coverage` asks
    "does every Rust function have a parity test"; it never asked "does every Python
    function have a Rust port", which is how an entire missing model (JOS3) stayed
    invisible while the build was green.
    """
    try:
        api = python_public_api()
    except ImportError as exc:
        print(
            f"error: could not import pythermalcomfort, so the Python-to-Rust "
            f"completeness check cannot run: {exc}\n"
            f"Set it up with: make setup-parity",
            file=sys.stderr,
        )
        return True, []

    expected = os.environ.get("PTC_VERSION")
    if expected:
        import pythermalcomfort

        actual = pythermalcomfort.__version__
        if actual != expected:
            print(
                f"error: comparing against pythermalcomfort {actual}, but this crate "
                f"ports {expected}. The completeness check would be inventorying the "
                f"wrong API.\nFix with: make setup-parity",
                file=sys.stderr,
            )
            return True, []

    rust = public_rust_identifiers()
    unported = sorted(
        name
        for name in api
        if name not in PYTHON_EXEMPT
        and PYTHON_TO_RUST.get(name, name) not in rust
    )

    failed = False

    stale = sorted(
        n for n in (PYTHON_EXEMPT | PYTHON_NOT_PORTED) if n not in api
    )
    if stale:
        failed = True
        print(
            "Stale entries in PYTHON_EXEMPT/PYTHON_NOT_PORTED (no longer in "
            "pythermalcomfort, remove them):",
            file=sys.stderr,
        )
        for name in stale:
            print(f"  - {name}", file=sys.stderr)

    resolved = sorted(n for n in PYTHON_NOT_PORTED if n in api and n not in unported)
    if resolved:
        failed = True
        print(
            f"\n{len(resolved)} pythermalcomfort name(s) now have a Rust port but are "
            f"still listed in PYTHON_NOT_PORTED. Remove them:\n",
            file=sys.stderr,
        )
        for name in resolved:
            print(f"  - {name}", file=sys.stderr)

    new_gaps = sorted(set(unported) - PYTHON_NOT_PORTED.keys())
    if new_gaps:
        failed = True
        print(
            f"\n{len(new_gaps)} pythermalcomfort name(s) have no Rust counterpart:\n",
            file=sys.stderr,
        )
        for name in new_gaps:
            print(f"  - {name} (pythermalcomfort.{api[name]})", file=sys.stderr)
        print(
            "\nPort it, or add it to PYTHON_TO_RUST if the port spells it differently, "
            "or to PYTHON_EXEMPT with a reason if it needs no counterpart.\n",
            file=sys.stderr,
        )

    outstanding = sorted(set(unported) & PYTHON_NOT_PORTED.keys())
    return failed, outstanding


def main() -> int:
    missing_files = [p for p in PARITY_TEST_FILES if not p.exists()]
    if missing_files:
        for path in missing_files:
            print(f"error: {path} not found", file=sys.stderr)
        return 1

    tests = "\n".join(
        executable_test_source(p.read_text()) for p in PARITY_TEST_FILES
    )
    functions = public_function_names()

    def is_tested(name: str) -> bool:
        # Require a call, not a mention. A bare-word match treats prose as evidence:
        # `at` matched 28 times in comments and assertion messages - including this
        # checker's own rationale - so `at()` counted as covered with its only parity
        # test deleted.
        return bool(re.search(rf"\b{re.escape(name)}\s*\(", tests))

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
            f"{' or '.join(str(p.relative_to(REPO)) for p in PARITY_TEST_FILES)}:\n",
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

    # The other direction: does every pythermalcomfort name have a Rust port?
    py_failed, py_outstanding = check_python_direction()
    failed = failed or py_failed

    if failed:
        return 1

    tested = len(functions) - len(EXEMPT) - len(outstanding)
    print(f"✓ {tested} public functions have parity tests, {len(EXEMPT)} exempt")
    if outstanding:
        print(f"  {len(outstanding)} known gap(s) outstanding: {', '.join(outstanding)}")
    print("✓ every pythermalcomfort name has a Rust port or a recorded reason")
    if py_outstanding:
        print(f"  {len(py_outstanding)} not ported yet: {', '.join(py_outstanding)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
