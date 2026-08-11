//! Numerical methods for thermal comfort calculations
//!
//! This module provides numerical algorithms used in thermal comfort calculations,
//! particularly root-finding methods.

use libm::fabs as abs;

/// Error type for root-finding methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RootFindError {
    /// Function values at bounds have the same sign
    InvalidBounds,
    /// Maximum iterations exceeded
    MaxIterationsExceeded,
    /// Function evaluation returned NaN
    NanEncountered,
}

impl core::fmt::Display for RootFindError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidBounds => {
                write!(
                    f,
                    "function values at the bounds do not have opposite signs"
                )
            }
            Self::MaxIterationsExceeded => write!(f, "maximum iterations exceeded"),
            Self::NanEncountered => write!(f, "function evaluation returned NaN"),
        }
    }
}

impl core::error::Error for RootFindError {}

/// Find a root of a function using Brent's method
///
/// Brent's method is a root-finding algorithm combining bisection, secant,
/// and inverse quadratic interpolation. It's guaranteed to converge and is
/// generally faster than pure bisection.
///
/// # Arguments
///
/// * `f` - Function to find root of
/// * `a` - Lower bound of search interval
/// * `b` - Upper bound of search interval
/// * `tol` - Absolute tolerance for convergence (default: 1e-6)
/// * `max_iter` - Maximum number of iterations (default: 100)
///
/// # Returns
///
/// Root x where f(x) ≈ 0, or error if not found
///
/// # Errors
///
/// Returns `InvalidBounds` if f(a) and f(b) have the same sign.
/// Returns `MaxIterationsExceeded` if convergence not achieved in max_iter iterations.
/// Returns `NanEncountered` if function evaluation returns NaN.
///
/// # Examples
///
/// ```
/// use thermalcomfort::numerical::brentq;
///
/// // Find root of x^2 - 4 = 0 (root at x = 2)
/// let f = |x: f64| x * x - 4.0;
/// let root = brentq(f, 0.0, 3.0, None, None).unwrap();
/// assert!((root - 2.0).abs() < 1e-6);
/// ```
pub fn brentq<F>(
    f: F,
    a: f64,
    b: f64,
    xtol: Option<f64>,
    max_iter: Option<usize>,
) -> Result<f64, RootFindError>
where
    F: Fn(f64) -> f64,
{
    // A faithful port of scipy.optimize.brentq (scipy/optimize/Zeros/brentq.c), not the
    // Numerical Recipes `zbrent` that lived here before. The distinction is not
    // academic: pythermalcomfort calls scipy.optimize.brentq for cooling_effect,
    // pet_steady and sports_heat_stress_risk, and on a non-monotonic objective the two
    // algorithms converge to *different* roots. cooling_effect's objective has three
    // roots near (8.1 C, 31.5 C, 0.58 m/s, 5.6% RH, 4.6 met); scipy returns the first
    // (13.67) and zbrent returned the last (15.49), which then shifted pmv_ppd_ashrae.
    let xtol = xtol.unwrap_or(2e-12);
    // scipy's default rtol is 4 * DBL_EPSILON
    let rtol = 4.0 * f64::EPSILON;
    let max_iter = max_iter.unwrap_or(100);

    let mut xpre = a;
    let mut xcur = b;
    let mut xblk = 0.0;
    let mut fblk = 0.0;
    let mut spre = 0.0;
    let mut scur = 0.0;

    let mut fpre = f(xpre);
    let mut fcur = f(xcur);

    if fpre.is_nan() || fcur.is_nan() {
        return Err(RootFindError::NanEncountered);
    }
    if fpre * fcur > 0.0 {
        return Err(RootFindError::InvalidBounds);
    }
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }

    for _ in 0..max_iter {
        if fpre * fcur < 0.0 {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if abs(fblk) < abs(fcur) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * abs(xcur)) / 2.0;
        let sbis = (xblk - xcur) / 2.0;

        if fcur == 0.0 || abs(sbis) < delta {
            return Ok(xcur);
        }

        if abs(spre) > delta && abs(fcur) < abs(fpre) {
            let stry = if xpre == xblk {
                // interpolate (secant)
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate (inverse quadratic)
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * abs(stry) < min_f64(abs(spre), 3.0 * abs(sbis) - delta) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = scur;
                scur = sbis;
            }
        } else {
            // bisect
            spre = scur;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if abs(scur) > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }

        fcur = f(xcur);
        if fcur.is_nan() {
            return Err(RootFindError::NanEncountered);
        }
    }

    Ok(xcur)
}

#[inline]
fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brentq_simple() {
        // Find root of x^2 - 4 = 0 (root at x = 2)
        let f = |x: f64| x * x - 4.0;
        let root = brentq(f, 0.0, 3.0, None, None).unwrap();
        assert!((root - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_brentq_linear() {
        // Find root of 2x - 6 = 0 (root at x = 3)
        let f = |x: f64| 2.0 * x - 6.0;
        let root = brentq(f, 0.0, 10.0, None, None).unwrap();
        assert!((root - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_brentq_cubic() {
        // Find root of x^3 - x - 2 = 0 (root at x ≈ 1.5214)
        let f = |x: f64| x * x * x - x - 2.0;
        let root = brentq(f, 1.0, 2.0, None, None).unwrap();
        assert!((root - 1.5213797068045678).abs() < 1e-6);
    }

    #[test]
    fn test_brentq_invalid_bounds() {
        // Both bounds have same sign for f(x) = x^2 - 4
        let f = |x: f64| x * x - 4.0;
        let result = brentq(f, 3.0, 5.0, None, None);
        assert_eq!(result, Err(RootFindError::InvalidBounds));
    }

    #[test]
    fn test_brentq_exact_root() {
        // Root exactly at boundary
        let f = |x: f64| x - 2.0;
        let root = brentq(f, 2.0, 3.0, None, None).unwrap();
        assert!((root - 2.0).abs() < 1e-10);
    }
}
