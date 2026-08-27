# DECISION NEEDED: three more models accept postures they cannot compute

**Same question already answered once for JOS3, now for its siblings.**

> `set_tmp`, `two_nodes_gagge` and `use_fans_heatwaves` take the crate-wide 7-variant
> `utilities::Posture`. Upstream accepts only `sitting`/`standing` and raises `ValueError`
> for the rest. Should these fields narrow to a 2-variant enum, as JOS3's did in 8592d90?

## The divergence

```
set_tmp(..., position = Posture::Sedentary)   -> this crate: returns a number
set_tmp(..., position = "sedentary")          -> pythermalcomfort: ValueError
```

Confirmed for all five of `sedentary`, `reclining`, `lying`, `supine`, `crouching`:
`position must be one of ['sitting', 'standing', 'standing, forced convection']`.

Internally the models branch only on `Posture::Sitting => 0.7, _ => 0.77`, so the five
unsupported variants silently take the standing arm and produce a plausible-looking result
for an input upstream rejects outright.

## Why it was not caught

The differential sweep cannot reach it: driving Python with those strings raises, so the
sweep can only ever cover the two variants upstream accepts. The gap is in the *type*, not
in the domain — exactly the shape the JOS3 decision addressed.

## Precedent

8592d90 narrowed `Jos3Conditions.posture` to a 3-variant `Jos3Posture` on the user's
instruction: *"correct by construction — with compile-time enforcement is the goal."*
`PhsPosture` is the same pattern. The variant sets differ per model and none is a subset
of the others, which is why a single shared enum cannot be right for all of them.

## Options

- **A — narrow all three** to a 2-variant enum (`Standing`, `Sitting`), mirroring
  `Jos3Posture`/`PhsPosture`. Makes the invalid case unrepresentable. Note
  `two_nodes_gagge` additionally has upstream's `"standing, forced convection"` value,
  which `pet::Posture` already models — check whether it belongs here too.
- **B — leave it**, and document that these models accept more postures than upstream and
  answer where upstream refuses.

**Recommendation: A**, for consistency with the decision already taken on JOS3.

## Not in scope

Deliberately not done as part of the Gap 2/3 test work, which the user gated ahead of
functional changes. This is a public API change and wants its own commit.
