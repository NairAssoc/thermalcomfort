# Lessons from porting pythermalcomfort

Each was expensive to learn and none is obvious from the code.

**One Rust helper standing in for two Python functions is this port's most productive bug
shape.** It has happened three times, and each instance was invisible until a randomised sweep
reached the corner where the two disagree:

- `libm::fmin` for Python's `min`. Three different semantics: `np.minimum` always propagates
  NaN, `libm::fmin` never does, and CPython's builtin propagates only when NaN is the *first*
  argument. Fixed with four named helpers in `utilities.rs`. A blanket "always propagate"
  replacement would have broken a third of the sites in the other direction.
- One `round_to` for two rounding rules. `round_to` matches `np.round`; CPython's builtin
  `round(x, n)` rounds the exact decimal expansion and differs on 89,262 of 200,000 tie-grid
  values. **Which one applies is decided by the argument type upstream, not by the name of
  the call**: `round()` on a `numpy.float64` dispatches to numpy's rule, on a plain `float`
  to CPython's, and inside a numba `@jit` to numpy's again.
- The `Temperature` newtype round-trip. `from_celsius(x).as_celsius()` loses the last ULP,
  which is harmless until a model walks a value onto an inclusive bound and depends on
  crossing it. JOS3 does exactly that, and it cost 2 °C on every set point for one subject
  in 3000. **Convert once at the public boundary; never re-wrap an `f64` mid-calculation.**

**A test can be shaped so it cannot fail.** Three examples, all of which passed for months: a parity test comparing one steady-state point at `epsilon = 2.0` against
a 2.17 °C error; a sweep that rounded *Python's* values to match Rust's before comparing,
making the reference a value pythermalcomfort never produces; and 53 JOS3 unit tests that
passed against an API no external caller could invoke, because they lived inside the module
and saw private items through `use super::*`. Doctests and the parity suite compile as
separate crates, which is what eventually caught the last one.

**"Swept N samples, found nothing" bounds what the sweep could reach, not what is true.**
The `Temperature` round-trip was closed as not reproducible on exactly that evidence. The
evidence was sound; the conclusion did not survive a model being added later that moved the
boundary the sweep had implicitly assumed nothing reached.

**Check whether upstream rounds before choosing a tolerance.** JOS3 rounds its outputs to
2 dp, so a short sweep run cannot see drift below 0.005 and passes vacuously. And when a
sweep *does* fail on a rounded field, distinguish a real defect from chaos by perturbing
**Python against Python** by one ULP: if upstream's own series moves the same way, no
implementation can match it there.

**Two conversion factors for one unit is the same bug as two helpers for one function.**
`p_sat_torr` built its `Pressure` with a hand-written 133.322 Pa/torr while `Pressure`
converts back with 101325/760, so a torr value round-tripped through the type came back
2.8e-6 too small — enough to move the Ji model's first minute by 1.5e-9 °C and fail a
fixture upstream pins at 1e-12. **Build a typed quantity with the constructor for the unit
you have, and read it back with the matching accessor; never inline the factor.** The
sweep had been comparing `as_pascals() / 133.322`, which cancels the error exactly, so it
could not see it.
