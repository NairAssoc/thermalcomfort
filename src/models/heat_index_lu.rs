//! Heat Index calculation using Lu and Romps (2022) model
//!
//! Advanced heat index model based on thermodynamic and thermoregulatory principles.

use measurements::{Temperature, Humidity};

/// Calculate Heat Index using Lu and Romps (2022) model
///
/// A physics-based model that accounts for thermodynamic and thermoregulatory
/// mechanisms to estimate apparent temperature under heat stress conditions.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
///
/// # Returns
///
/// Heat Index [°C]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::heat_index_lu::heat_index_lu;
/// use thermalcomfort::{Temperature, Humidity};
///
/// let hi = heat_index_lu(Temperature::from_celsius(25.0), Humidity::from_percent(50.0));
/// assert!((hi - 25.0).abs() < 1.0);
/// ```
///
/// # References
///
/// - Lu and Romps (2022)
pub fn heat_index_lu(dry_bulb_temp: Temperature, relative_humidity: Humidity) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let tdb_k = dry_bulb_celsius + 273.15;
    let rh_frac = relative_humidity.as_percent() / 100.0;

    let hi_k = lu_heat_index_core(tdb_k, rh_frac);
    let hi = hi_k - 273.15;

    libm::round(hi * 10.0) / 10.0
}

fn lu_heat_index_core(tdb: f64, rh: f64) -> f64 {
    // Thermodynamic parameters
    const T_C_K: f64 = 273.16;
    const P_TRIPLE_POINT: f64 = 611.65;
    const E0V: f64 = 2.3740e6;
    const E0S: f64 = 0.3337e6;
    const RGASA: f64 = 287.04;
    const RGASV: f64 = 461.0;
    const CVA: f64 = 719.0;
    const CVV: f64 = 1418.0;
    const CVL: f64 = 4119.0;
    const CVS: f64 = 1861.0;
    const CPA: f64 = CVA + RGASA;
    const CPV: f64 = CVV + RGASV;

    // Thermoregulatory parameters
    const SIGMA: f64 = 5.67e-8;
    const EPSILON: f64 = 0.97;
    const MASS: f64 = 83.6;
    const HEIGHT: f64 = 1.69;
    const CPC: f64 = 3492.0;
    const R: f64 = 124.0;
    const Q: f64 = 180.0;
    const PHI_SALT: f64 = 0.9;
    const T_CR: f64 = 310.0;
    const P: f64 = 1.013e5;
    const ETA: f64 = 1.43e-6;
    const PA0: f64 = 1.6e3;
    const ZA: f64 = 60.6 / 17.4;
    const ZA_BAR: f64 = 60.6 / 11.6;
    const ZA_UN: f64 = 60.6 / 12.3;
    const TOL: f64 = 1e-8;
    const TOL_T: f64 = 1e-8;
    const MAX_ITER: usize = 100;

    let area = 0.202 * libm::pow(MASS, 0.425) * libm::pow(HEIGHT, 0.725);
    let hc_core = MASS * CPC / area;

    let pv_star = |t: f64| -> f64 {
        if t == 0.0 {
            return 0.0;
        }
        if t < T_C_K {
            P_TRIPLE_POINT
                * libm::pow(t / T_C_K, (CPV - CVS) / RGASV)
                * libm::exp((E0V + E0S - (CVV - CVS) * T_C_K) / RGASV * (1.0 / T_C_K - 1.0 / t))
        } else {
            P_TRIPLE_POINT
                * libm::pow(t / T_C_K, (CPV - CVL) / RGASV)
                * libm::exp((E0V - (CVV - CVL) * T_C_K) / RGASV * (1.0 / T_C_K - 1.0 / t))
        }
    };

    let latent_heat_vap = |t: f64| -> f64 {
        E0V + (CVV - CVL) * (t - T_C_K) + RGASV * t
    };

    let p_cr = PHI_SALT * pv_star(T_CR);
    let lat_heat = latent_heat_vap(310.0);

    let qv = |ta: f64, pa: f64| -> f64 {
        ETA * Q * (CPA * (T_CR - ta) + lat_heat * RGASA / (P * RGASV) * (p_cr - pa))
    };

    let zs = |rs: f64| -> f64 {
        if (rs - 0.0387).abs() < 1e-10 {
            52.1
        } else {
            6.0e8 * libm::pow(rs, 5.0)
        }
    };

    let ra = |ts: f64, ta: f64| -> f64 {
        let hc = 17.4;
        let phi_rad = 0.85;
        let hr = EPSILON * phi_rad * SIGMA * (ts * ts + ta * ta) * (ts + ta);
        1.0 / (hc + hr)
    };

    let ra_bar = |tf: f64, ta: f64| -> f64 {
        let hc = 11.6;
        let phi_rad = 0.79;
        let hr = EPSILON * phi_rad * SIGMA * (tf * tf + ta * ta) * (tf + ta);
        1.0 / (hc + hr)
    };

    let ra_un = |ts: f64, ta: f64| -> f64 {
        let hc = 12.3;
        let phi_rad = 0.80;
        let hr = EPSILON * phi_rad * SIGMA * (ts * ts + ta * ta) * (ts + ta);
        1.0 / (hc + hr)
    };

    let solve = |f: &dyn Fn(f64) -> f64, x1: f64, x2: f64, tol: f64, max_iter: usize| -> f64 {
        let mut a = x1;
        let mut b = x2;
        let fa = f(a);
        let mut fb = f(b);

        if fa * fb > 0.0 {
            return (a + b) / 2.0; // fallback
        }

        for _ in 0..max_iter {
            let c = (a + b) / 2.0;
            let fc = f(c);

            if fb * fc > 0.0 {
                b = c;
                fb = fc;
            } else {
                a = c;
            }

            if libm::fabs(a - b) < tol {
                return c;
            }
        }
        (a + b) / 2.0
    };

    let find_eq_var = |ta: f64, _rh: f64| -> (usize, f64, f64, f64, f64) {
        let pa = _rh * pv_star(ta);
        let rs = 0.0387;
        let phi = 0.84;
        let d_tc_dt = 0.0;

        let m = (p_cr - pa) / (zs(rs) + ZA);
        let m_bar = (p_cr - pa) / (zs(rs) + ZA_BAR);

        let ts = solve(
            &|ts: f64| -> f64 {
                (ts - ta) / ra(ts, ta) + (p_cr - pa) / (zs(rs) + ZA) - (T_CR - ts) / rs
            },
            libm::fmax(0.0, libm::fmin(T_CR, ta) - rs * libm::fabs(m)),
            libm::fmax(T_CR, ta) + rs * libm::fabs(m),
            TOL,
            MAX_ITER,
        );

        let tf = solve(
            &|tf: f64| -> f64 {
                (tf - ta) / ra_bar(tf, ta) + (p_cr - pa) / (zs(rs) + ZA_BAR) - (T_CR - tf) / rs
            },
            libm::fmax(0.0, libm::fmin(T_CR, ta) - rs * libm::fabs(m_bar)),
            libm::fmax(T_CR, ta) + rs * libm::fabs(m_bar),
            TOL,
            MAX_ITER,
        );

        let flux1 = Q - qv(ta, pa) - (1.0 - phi) * (T_CR - ts) / rs;
        let flux2 = Q - qv(ta, pa) - (1.0 - phi) * (T_CR - ts) / rs - phi * (T_CR - tf) / rs;

        if flux1 <= 0.0 {
            // Region I
            let phi_new = 1.0 - (Q - qv(ta, pa)) * rs / (T_CR - ts);
            (1, phi_new, f64::INFINITY, rs, d_tc_dt)
        } else if flux2 <= 0.0 {
            // Region II&III
            let ts_bar = T_CR - (Q - qv(ta, pa)) * rs / phi + (1.0 / phi - 1.0) * (T_CR - ts);
            let tf_new = solve(
                &|tf: f64| -> f64 {
                    (tf - ta) / ra_bar(tf, ta)
                        + (p_cr - pa) * (tf - ta)
                            / ((zs(rs) + ZA_BAR) * (tf - ta) + R * ra_bar(tf, ta) * (ts_bar - tf))
                        - (T_CR - ts_bar) / rs
                },
                ta,
                ts_bar,
                TOL,
                MAX_ITER,
            );
            let rf = ra_bar(tf_new, ta) * (ts_bar - tf_new) / (tf_new - ta);
            (2, phi, rf, rs, d_tc_dt)
        } else {
            // Region IV,V,VI
            let flux3 = Q - qv(ta, pa) - (T_CR - ta) / ra_un(T_CR, ta)
                - (PHI_SALT * pv_star(T_CR) - pa) / ZA_UN;

            if flux3 < 0.0 {
                // Region IV,V
                let ts_new = solve(
                    &|ts: f64| -> f64 {
                        let rs_local = (T_CR - ts) / (Q - qv(ta, pa));
                        (ts - ta) / ra_un(ts, ta) + (p_cr - pa) / (zs(rs_local) + ZA_UN) - (Q - qv(ta, pa))
                    },
                    0.0,
                    T_CR,
                    TOL,
                    MAX_ITER,
                );
                let rs_new = (T_CR - ts_new) / (Q - qv(ta, pa));
                let ps = p_cr - (p_cr - pa) * zs(rs_new) / (zs(rs_new) + ZA_UN);

                if ps > PHI_SALT * pv_star(ts_new) {
                    // Region V
                    let ts_final = solve(
                        &|ts: f64| -> f64 {
                            (ts - ta) / ra_un(ts, ta) + (PHI_SALT * pv_star(ts) - pa) / ZA_UN - (Q - qv(ta, pa))
                        },
                        0.0,
                        T_CR,
                        TOL,
                        MAX_ITER,
                    );
                    let rs_final = (T_CR - ts_final) / (Q - qv(ta, pa));
                    (4, phi, 0.0, rs_final, d_tc_dt)
                } else {
                    // Region IV
                    (3, phi, 0.0, rs_new, d_tc_dt)
                }
            } else {
                // Region VI
                let d_tc_dt_new = (1.0 / hc_core) * flux3;
                (5, phi, 0.0, 0.0, d_tc_dt_new)
            }
        }
    };

    let find_t = |eq_var_type: usize, eq_var: f64| -> f64 {
        match eq_var_type {
            1 => {
                // phi
                solve(
                    &|t: f64| -> f64 {
                        let (_, phi, _, _, _) = find_eq_var(t, 1.0);
                        phi - eq_var
                    },
                    0.0,
                    240.0,
                    TOL_T,
                    MAX_ITER,
                )
            }
            2 => {
                // rf
                solve(
                    &|t: f64| -> f64 {
                        let rh_local = libm::fmin(1.0, PA0 / pv_star(t));
                        let (_, _, rf, _, _) = find_eq_var(t, rh_local);
                        rf - eq_var
                    },
                    230.0,
                    300.0,
                    TOL_T,
                    MAX_ITER,
                )
            }
            3 | 4 => {
                // rs or rs*
                solve(
                    &|t: f64| -> f64 {
                        let rh_local = PA0 / pv_star(t);
                        let (_, _, _, rs, _) = find_eq_var(t, rh_local);
                        rs - eq_var
                    },
                    295.0,
                    350.0,
                    TOL_T,
                    MAX_ITER,
                )
            }
            _ => {
                // d_tc_dt
                solve(
                    &|t: f64| -> f64 {
                        let rh_local = PA0 / pv_star(t);
                        let (_, _, _, _, d_tc_dt) = find_eq_var(t, rh_local);
                        d_tc_dt - eq_var
                    },
                    340.0,
                    1000.0,
                    TOL_T,
                    MAX_ITER,
                )
            }
        }
    };

    if tdb == 0.0 {
        return 0.0;
    }

    let (eq_var_type, phi, rf, rs, d_tc_dt) = find_eq_var(tdb, rh);

    let eq_var = match eq_var_type {
        1 => phi,
        2 => rf,
        3 | 4 => rs,
        _ => d_tc_dt,
    };

    find_t(eq_var_type, eq_var)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_index_lu() {
        let hi = heat_index_lu(Temperature::from_celsius(25.0), Humidity::from_percent(50.0));
        // Should be close to 25.9°C
        assert!(hi > 24.0 && hi < 27.0);
    }

    #[test]
    fn test_heat_index_lu_high_temp() {
        let hi = heat_index_lu(Temperature::from_celsius(35.0), Humidity::from_percent(70.0));
        // Should be significantly higher than air temperature
        assert!(hi > 35.0);
    }
}
