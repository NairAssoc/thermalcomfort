//! Numerical methods for thermal comfort calculations
//!
//! This module provides numerical algorithms used in thermal comfort calculations,
//! particularly root-finding methods.

use libm::{fabs as abs, copysign};

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
    mut a: f64,
    mut b: f64,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> Result<f64, RootFindError>
where
    F: Fn(f64) -> f64,
{
    let tol = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(100);

    let mut fa = f(a);
    let mut fb = f(b);

    if fa.is_nan() || fb.is_nan() {
        return Err(RootFindError::NanEncountered);
    }

    if fa * fb > 0.0 {
        return Err(RootFindError::InvalidBounds);
    }

    if abs(fa) < abs(fb) {
        // Swap a and b
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    for _ in 0..max_iter {
        if abs(fc) < abs(fb) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol1 = 2.0 * f64::EPSILON * abs(b) + 0.5 * tol;
        let xm = 0.5 * (c - b);

        if abs(xm) <= tol1 || fb == 0.0 {
            return Ok(b);
        }

        if abs(e) >= tol1 && abs(fa) > abs(fb) {
            // Attempt inverse quadratic interpolation
            let s = fb / fa;
            let (p, q) = if a == c {
                // Linear interpolation (secant method)
                let p = 2.0 * xm * s;
                let q = 1.0 - s;
                (p, q)
            } else {
                // Inverse quadratic interpolation
                let q = fa / fc;
                let r = fb / fc;
                let p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                let q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                (p, q)
            };

            let p = if p > 0.0 { -q } else { p };
            let q = if p > 0.0 { abs(q) } else { q };

            let p_abs = abs(p);
            let min1 = 3.0 * xm * q - abs(tol1 * q);
            let min2 = abs(e * q);

            if 2.0 * p_abs < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                // Interpolation failed, use bisection
                d = xm;
                e = d;
            }
        } else {
            // Bounds decreasing too slowly, use bisection
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;

        if abs(d) > tol1 {
            b += d;
        } else {
            b += copysign(tol1, xm);
        }

        fb = f(b);

        if fb.is_nan() {
            return Err(RootFindError::NanEncountered);
        }

        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
    }

    Err(RootFindError::MaxIterationsExceeded)
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
