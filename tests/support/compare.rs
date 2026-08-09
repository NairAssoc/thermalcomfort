//! Field-by-field comparison of a Rust result against a Python one.

/// How a NaN on one side should be treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanPolicy {
    /// NaN on one side and a number on the other is a failure. The default.
    MustMatch,
    /// Rust may return NaN where Python returns a number. Reserved for the documented
    /// `no_std` PET solver deviation — do not reach for it to silence anything else,
    /// because it hides exactly the class of bug this sweep exists to find.
    RustMayBeNan,
}

#[derive(Debug, Clone)]
pub struct FieldCmp {
    pub name: &'static str,
    pub tol: f64,
    /// Optional relative tolerance, as a fraction of the larger magnitude.
    ///
    /// For quantities built by integrating over many timesteps (PHS runs 480 minutes),
    /// per-step float differences accumulate in proportion to the result, so an absolute
    /// tolerance is the wrong instrument: it is either too tight at large magnitudes or
    /// far too slack at small ones. A field passes if EITHER bound is met.
    pub rel: Option<f64>,
    pub nan: NanPolicy,
}

impl FieldCmp {
    pub fn new(name: &'static str, tol: f64) -> Self {
        Self {
            name,
            tol,
            rel: None,
            nan: NanPolicy::MustMatch,
        }
    }

    /// Also accept a relative difference of `rel` (e.g. `1e-4` for 0.01%).
    pub fn rel(mut self, rel: f64) -> Self {
        self.rel = Some(rel);
        self
    }

    pub fn nan(mut self, policy: NanPolicy) -> Self {
        self.nan = policy;
        self
    }
}

/// Compare one field. The error names the field and both values so a sweep failure is
/// actionable without re-running under a debugger.
pub fn compare_field(field: &FieldCmp, rust: f64, py: f64) -> Result<(), String> {
    match (rust.is_nan(), py.is_nan()) {
        (true, true) => Ok(()),
        (true, false) => {
            if field.nan == NanPolicy::RustMayBeNan {
                Ok(())
            } else {
                Err(format!("{}: Rust NaN, Python {py}", field.name))
            }
        }
        (false, true) => Err(format!("{}: Rust {rust}, Python NaN", field.name)),
        (false, false) => {
            let delta = (rust - py).abs();
            let rel_ok = field
                .rel
                .is_some_and(|r| delta <= r * rust.abs().max(py.abs()));
            if delta <= field.tol || rel_ok {
                Ok(())
            } else {
                Err(format!(
                    "{}: Rust {rust}, Python {py} (delta {delta:.6}, tol {}{})",
                    field.name,
                    field.tol,
                    match field.rel {
                        Some(r) => format!(", rel {r:e}"),
                        None => String::new(),
                    }
                ))
            }
        }
    }
}
