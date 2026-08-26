# DECISION NEEDED: `Temperature` cannot represent a plain decimal, and it changes answers

**One question, and it is an API-level one, which is why it is not being answered here.**

> `measurements::Temperature` stores kelvin. Should the crate keep it on the public
> surface, given that it makes some results disagree with pythermalcomfort?

## The divergence, reproducible today

```
adaptive_en(tdb=24.1, tr=24.1, t_running_mean=10, v=0.1)
  pythermalcomfort  -> acceptability_cat_i = True
  this crate        -> acceptability_cat_i = false
```

`Temperature::from_celsius(24.1).as_celsius()` is `24.100000000000023`, not
`24.1`. EN's category-I upper bound at this running mean is `24.100000000000001`.
So the operative temperature lands *above* the bound and the flag flips.

**The banding logic is correct.** Given the same round-tripped float, Python agrees:

```
adaptive_en(tdb=24.100000000000023, ...) -> acceptability_cat_i = False
```

The two implementations agree on the rule and disagree on what "24.1 °C" is.

## Why the existing rule does not cover this

The rule this repo already carries — *never re-wrap a plain `f64` into a newtype
mid-calculation; convert once at the public boundary and calculate in `f64`* — came out
of the JOS3 ISO-7730-limit bug (28adfa8) and is sound. It does not help here, because
**the boundary itself is lossy.** There is no interior re-wrap left to remove: the caller
hands in a `Temperature`, and it is already wrong by ~2e-14 before the model starts.

Every decimal that is not binary-representable is affected. `adaptive_ashrae` passes the
same edge test only by luck: it rounds `t_cmf` before applying the ±3.5 / ±2.5 offsets, so
its bounds are 20.5 and 27.5, which survive the round trip exactly.

## Scope

Unknown, and worth measuring before choosing. Any discrete output compared against a
threshold derived from a temperature is a candidate — the adaptive acceptability flags are
simply where an edge test happened to look first. The continuous outputs are unaffected in
any way a user would notice (~2e-14 °C).

## Options

- **A — leave it.** Zero work. The crate keeps one clean newtype API and accepts that
  results can differ from upstream in the last band near an edge. Document it.
- **B — store Celsius internally** in the wrapper, so `from_celsius`/`as_celsius` is the
  identity and kelvin becomes the converted-on-demand unit. Fixes the whole class at once.
  Requires owning the type rather than re-exporting `measurements`'.
- **C — take plain `f64` °C on the public API** for the models where a threshold
  comparison is involved. Largest blast radius, undoes the typed-quantity API pass this
  branch just did. Not recommended.

**This branch is already a major bump (3.9.8 → 4.4.2), so B is in scope if wanted.**

## Not in scope

`operative_temperature_celsius` (added 2026-08-26) is a separate, already-done fix: it
removes an *internal* re-wrap in `adaptive_ashrae`/`adaptive_en` where the operative
temperature was computed in `f64`, wrapped into a `Temperature`, and immediately
unwrapped. Correct per the existing rule, and it does not address anything above.
