//! Specialty thermal comfort models and indices
//!
//! This module contains specialized models for specific comfort assessment scenarios.

use crate::models::pmv::{PmvPpdAshraeOptions, PmvPpdInputs, pmv_ppd_ashrae};
use crate::{ClothingInsulation, MetabolicRate, TemperatureDelta};
use measurements::{Humidity, Speed, Temperature};

/// The comfort inputs to [`ankle_draft`].
///
/// `tdb` and `tr` are consecutive [`Temperature`]s; naming
/// every field forecloses a silent transposition between them.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnkleDraftInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Relative air speed (must be < 0.2 m/s for this equation to apply)
    pub vr: Speed,
    /// Relative humidity
    pub rh: Humidity,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation
    pub clo: ClothingInsulation,
    /// Air speed at 0.1 m above the floor
    pub v_ankle: Speed,
}

/// Optional parameters for [`ankle_draft`], with pythermalcomfort's defaults.
///
/// Upstream also accepts `units: {'SI', 'IP'}`, but that parameter only tells Python how
/// to interpret its raw floats; the [`Temperature`]/[`Speed`] newtypes in
/// [`AnkleDraftInputs`] already carry that information, so (as with `pmv_ppd_ashrae`) a
/// `units` flag has nothing left to do here and is not ported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnkleDraftOptions {
    /// If true, returns NaN/false when any input is outside the ASHRAE 55
    /// applicability range: 10 ≤ tdb [°C] ≤ 40, 10 ≤ tr [°C] ≤ 40, 0 ≤ vr [m/s] ≤ 0.2,
    /// 1 ≤ met ≤ 4, 0 ≤ clo ≤ 1.5.
    pub limit_inputs: bool,
}

impl Default for AnkleDraftOptions {
    fn default() -> Self {
        Self { limit_inputs: true }
    }
}

/// Calculate percentage dissatisfied due to ankle draft
///
/// Calculates the percentage of thermally dissatisfied people with the ankle draft
/// (0.1 m) above floor level. Only applicable for vr < 0.2 m/s.
///
/// # Returns
///
/// Tuple of (ppd_ankle_draft %, acceptability bool)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::specialty::{ankle_draft, AnkleDraftInputs};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let (ppd, acceptable) = ankle_draft(
///     AnkleDraftInputs {
///         tdb: Temperature::from_celsius(23.0),
///         tr: Temperature::from_celsius(23.0),
///         vr: Speed::from_meters_per_second(0.1),
///         rh: Humidity::from_percent(45.0),
///         met: MetabolicRate::from_met(1.1),
///         clo: ClothingInsulation::from_clo(0.7),
///         v_ankle: Speed::from_meters_per_second(0.15),
///     },
///     Default::default(),
/// );
/// println!("PPD ankle draft: {:.1}%, Acceptable: {}", ppd, acceptable);
/// ```
///
/// # References
///
/// - Liu et al. (2017)
/// - ASHRAE 55-2023
pub fn ankle_draft(inputs: AnkleDraftInputs, options: AnkleDraftOptions) -> (f64, bool) {
    let AnkleDraftInputs {
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        v_ankle,
    } = inputs;
    let AnkleDraftOptions { limit_inputs } = options;

    // Calculate PMV value for use in ankle draft equation.
    // Matches pythermalcomfort behaviour: PMV is computed without input limits so the
    // outer limit_inputs flag governs the final return value.
    let pmv_result = pmv_ppd_ashrae(
        PmvPpdInputs {
            tdb,
            tr,
            vr,
            rh,
            met,
            clo,
        },
        PmvPpdAshraeOptions {
            limit_inputs: false,
            ..Default::default()
        },
    );
    let pmv = pmv_result.pmv; // Use PMV value directly, not TSV enum

    let ankle_speed = v_ankle.as_meters_per_second();

    // Calculate PPD for ankle draft using logistic function
    let exponent = -2.58 + 3.05 * ankle_speed - 1.06 * pmv;
    let ppd_ad = (libm::exp(exponent) / (1.0 + libm::exp(exponent))) * 100.0;
    let ppd_ad = crate::utilities::round_half_even(ppd_ad * 10.0) / 10.0;

    if limit_inputs && !ashrae55_ankle_inputs_valid(tdb, tr, vr, met, clo) {
        return (f64::NAN, false);
    }

    let acceptability = ppd_ad <= 20.0;

    (ppd_ad, acceptability)
}

fn ashrae55_ankle_inputs_valid(
    tdb: Temperature,
    tr: Temperature,
    vr: Speed,
    met: MetabolicRate,
    clo: ClothingInsulation,
) -> bool {
    let tdb = tdb.as_celsius();
    let tr = tr.as_celsius();
    let vr = vr.as_meters_per_second();
    let met = met.as_met();
    let clo = clo.as_clo();
    (10.0..=40.0).contains(&tdb)
        && (10.0..=40.0).contains(&tr)
        && (0.0..=0.2).contains(&vr)
        && (1.0..=4.0).contains(&met)
        && (0.0..=1.5).contains(&clo)
}

/// The comfort inputs to [`vertical_tmp_grad_ppd`].
///
/// `tdb` and `tr` are consecutive [`Temperature`]s; naming
/// every field forecloses a silent transposition between them.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalTmpGradPpdInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Relative air speed
    pub vr: Speed,
    /// Relative humidity
    pub rh: Humidity,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation
    pub clo: ClothingInsulation,
    /// Vertical temperature gradient between 1.1 m and 0.1 m (a [`TemperatureDelta`], so
    /// the unit is explicit rather than implied)
    pub vertical_tmp_grad: TemperatureDelta,
}

/// Optional parameters for [`vertical_tmp_grad_ppd`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalTmpGradPpdOptions {
    /// If true, rounds the returned PPD to one decimal place. Acceptability is always
    /// judged on the unrounded value, matching upstream.
    pub round_output: bool,
    /// If true, returns NaN/false when any input is outside the ASHRAE 55
    /// applicability range: 10 ≤ tdb [°C] ≤ 40, 10 ≤ tr [°C] ≤ 40, 0 ≤ vr [m/s] ≤ 0.2,
    /// 1 ≤ met ≤ 4, 0 ≤ clo ≤ 1.5.
    pub limit_inputs: bool,
}

impl Default for VerticalTmpGradPpdOptions {
    fn default() -> Self {
        Self {
            round_output: true,
            limit_inputs: true,
        }
    }
}

/// Calculate PPD for vertical air temperature gradient
///
/// Calculates the percentage of thermally dissatisfied people with a vertical
/// temperature gradient between feet and head.
///
/// # Returns
///
/// Tuple of (ppd %, acceptability bool). The PPD is ASHRAE 55's offset-logistic fit,
/// `max((x / (1 + x) - 0.345) * 100, 0)` with
/// `x = exp(0.13 (PMV - 1.91)² + 0.15 gradient - 1.6)`: the 34.5 % term is the baseline
/// share of dissatisfied occupants, so small gradients near PMV 0 give 0 %.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::specialty::{vertical_tmp_grad_ppd, VerticalTmpGradPpdInputs};
/// use thermalcomfort::{
///     Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation, TemperatureDelta,
/// };
///
/// let (ppd, acceptable) = vertical_tmp_grad_ppd(
///     VerticalTmpGradPpdInputs {
///         tdb: Temperature::from_celsius(25.0),
///         tr: Temperature::from_celsius(25.0),
///         vr: Speed::from_meters_per_second(0.1),
///         rh: Humidity::from_percent(50.0),
///         met: MetabolicRate::from_met(1.2),
///         clo: ClothingInsulation::from_clo(0.5),
///         vertical_tmp_grad: TemperatureDelta::from_celsius(2.0),
///     },
///     Default::default(),
/// );
/// println!("PPD vertical gradient: {:.1}%, Acceptable: {}", ppd, acceptable);
/// ```
///
/// # References
///
/// - ISO 7730:2005
/// - ASHRAE 55-2023
pub fn vertical_tmp_grad_ppd(
    inputs: VerticalTmpGradPpdInputs,
    options: VerticalTmpGradPpdOptions,
) -> (f64, bool) {
    let VerticalTmpGradPpdInputs {
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        vertical_tmp_grad,
    } = inputs;
    let VerticalTmpGradPpdOptions {
        round_output,
        limit_inputs,
    } = options;

    // Calculate PMV value for use in vertical temperature gradient equation.
    // Matches pythermalcomfort behaviour: PMV is computed without input limits so the
    // outer limit_inputs flag governs the final return value.
    let pmv_result = pmv_ppd_ashrae(
        PmvPpdInputs {
            tdb,
            tr,
            vr,
            rh,
            met,
            clo,
        },
        PmvPpdAshraeOptions {
            limit_inputs: false,
            ..Default::default()
        },
    );
    let pmv = pmv_result.pmv;
    let vertical_tmp_grad = vertical_tmp_grad.as_celsius();

    // PPD calculation for vertical temperature gradient using ASHRAE 55-2023 formula
    let numerator = libm::exp(0.13 * libm::pow(pmv - 1.91, 2.0) + 0.15 * vertical_tmp_grad - 1.6);
    // The model subtracts a 34.5 % baseline of dissatisfied occupants; below it the
    // dissatisfaction is zero, not negative. NaN propagates as upstream's np.maximum does.
    let ppd_vtg =
        crate::utilities::np_maximum((numerator / (1.0 + numerator) - 0.345) * 100.0, 0.0);
    // Acceptability is judged on the unrounded value, then the value is rounded.
    // (ankle_draft rounds first - the port had applied ankle_draft's ordering to both.)
    let acceptability_raw = ppd_vtg <= 5.0;
    // Upstream exposes this as `round_output`; rounding unconditionally, as this did
    // before, left the full-precision value unobtainable.
    let ppd_vtg = if round_output {
        crate::utilities::round_half_even(ppd_vtg * 10.0) / 10.0
    } else {
        ppd_vtg
    };

    if limit_inputs && !ashrae55_ankle_inputs_valid(tdb, tr, vr, met, clo) {
        return (f64::NAN, false);
    }

    let acceptability = acceptability_raw;

    (ppd_vtg, acceptability)
}

// ---------------------------------------------------------------------------
// Paths pythermalcomfort 4.6.0 moved elsewhere. Each forwards to its new home and is
// removed after two minor releases, as upstream's compatibility shims are.
// ---------------------------------------------------------------------------

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
pub type FSvvInputs = crate::environment::FSvvInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
#[inline]
pub fn f_svv(inputs: crate::environment::FSvvInputs) -> f64 {
    crate::environment::f_svv(inputs)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
#[inline]
pub fn transpose_sharp_altitude(
    sharp: measurements::Angle,
    altitude: measurements::Angle,
) -> (measurements::Angle, measurements::Angle) {
    crate::environment::transpose_sharp_altitude(sharp, altitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `AnkleDraftInputs` for a case, with options left at their defaults.
    fn ankle_inputs(
        tdb: f64,
        tr: f64,
        vr: f64,
        met: f64,
        clo: f64,
        v_ankle: f64,
    ) -> AnkleDraftInputs {
        AnkleDraftInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            vr: Speed::from_meters_per_second(vr),
            rh: Humidity::from_percent(50.0),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
            v_ankle: Speed::from_meters_per_second(v_ankle),
        }
    }

    /// `VerticalTmpGradPpdInputs` for a case, with options left at their defaults.
    fn vtg_inputs(
        tdb: f64,
        tr: f64,
        vr: f64,
        met: f64,
        clo: f64,
        grad: f64,
    ) -> VerticalTmpGradPpdInputs {
        VerticalTmpGradPpdInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            vr: Speed::from_meters_per_second(vr),
            rh: Humidity::from_percent(50.0),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
            vertical_tmp_grad: TemperatureDelta::from_celsius(grad),
        }
    }

    #[test]
    fn test_ankle_draft() {
        let (ppd, acceptable) = ankle_draft(
            ankle_inputs(25.0, 25.0, 0.1, 1.2, 0.5, 0.3),
            Default::default(),
        );
        assert!((0.0..=100.0).contains(&ppd));
        // High ankle draft velocity should cause dissatisfaction
        assert!(!acceptable || ppd <= 20.0);
    }

    #[test]
    fn test_ankle_draft_limit_inputs() {
        // Table-driven: one row per ASHRAE 55 applicability boundary.
        // For each row: limit_inputs=true → NaN/false; limit_inputs=false → numeric.
        let cases: &[(&str, f64, f64, f64, f64, f64)] = &[
            ("tdb_below_10", 5.0, 25.0, 0.1, 1.2, 0.5),
            ("tdb_above_40", 45.0, 25.0, 0.1, 1.2, 0.5),
            ("tr_below_10", 25.0, 5.0, 0.1, 1.2, 0.5),
            ("tr_above_40", 25.0, 45.0, 0.1, 1.2, 0.5),
            ("vr_above_0_2", 25.0, 25.0, 0.5, 1.2, 0.5),
            ("met_below_1", 25.0, 25.0, 0.1, 0.5, 0.5),
            ("met_above_4", 25.0, 25.0, 0.1, 5.0, 0.5),
            ("clo_above_1_5", 25.0, 25.0, 0.1, 1.2, 2.0),
        ];

        for &(label, tdb, tr, vr, met, clo) in cases {
            let (ppd, acceptable) = ankle_draft(
                ankle_inputs(tdb, tr, vr, met, clo, 0.15),
                AnkleDraftOptions { limit_inputs: true },
            );
            assert!(ppd.is_nan(), "{label}: expected NaN with limit_inputs=true");
            assert!(
                !acceptable,
                "{label}: expected !acceptable with limit_inputs=true"
            );

            let (ppd_unlimited, _) = ankle_draft(
                ankle_inputs(tdb, tr, vr, met, clo, 0.15),
                AnkleDraftOptions {
                    limit_inputs: false,
                },
            );
            assert!(
                !ppd_unlimited.is_nan(),
                "{label}: expected numeric PPD with limit_inputs=false"
            );
        }

        // Spot-check that a fully in-range input is not flagged when limits are on.
        let (ppd, _) = ankle_draft(
            ankle_inputs(25.0, 25.0, 0.1, 1.2, 0.5, 0.15),
            Default::default(),
        );
        assert!(
            !ppd.is_nan(),
            "in-range inputs should not be filtered by limit_inputs=true"
        );
    }

    #[test]
    fn test_vertical_tmp_grad_ppd() {
        let (ppd, _acceptable) = vertical_tmp_grad_ppd(
            vtg_inputs(25.0, 25.0, 0.1, 1.2, 0.5, 2.0),
            Default::default(),
        );
        // PPD can be negative for comfortable conditions (formula artifact)
        // but should be within reasonable range
        assert!((-50.0..=100.0).contains(&ppd));
    }

    #[test]
    fn test_vertical_tmp_grad_ppd_limit_inputs() {
        // Same boundary sweep as ankle_draft — the helper enforces identical limits.
        let cases: &[(&str, f64, f64, f64, f64, f64)] = &[
            ("tdb_below_10", 5.0, 25.0, 0.1, 1.2, 0.5),
            ("tdb_above_40", 45.0, 25.0, 0.1, 1.2, 0.5),
            ("tr_below_10", 25.0, 5.0, 0.1, 1.2, 0.5),
            ("tr_above_40", 25.0, 45.0, 0.1, 1.2, 0.5),
            ("vr_above_0_2", 25.0, 25.0, 0.5, 1.2, 0.5),
            ("met_below_1", 25.0, 25.0, 0.1, 0.5, 0.5),
            ("met_above_4", 25.0, 25.0, 0.1, 5.0, 0.5),
            ("clo_above_1_5", 25.0, 25.0, 0.1, 1.2, 2.0),
        ];

        for &(label, tdb, tr, vr, met, clo) in cases {
            let (ppd, acceptable) = vertical_tmp_grad_ppd(
                vtg_inputs(tdb, tr, vr, met, clo, 2.0),
                VerticalTmpGradPpdOptions {
                    round_output: true,
                    limit_inputs: true,
                },
            );
            assert!(ppd.is_nan(), "{label}: expected NaN with limit_inputs=true");
            assert!(
                !acceptable,
                "{label}: expected !acceptable with limit_inputs=true"
            );

            let (ppd_unlimited, _) = vertical_tmp_grad_ppd(
                vtg_inputs(tdb, tr, vr, met, clo, 2.0),
                VerticalTmpGradPpdOptions {
                    round_output: true,
                    limit_inputs: false,
                },
            );
            assert!(
                !ppd_unlimited.is_nan(),
                "{label}: expected numeric PPD with limit_inputs=false"
            );
        }

        // Spot-check that fully in-range inputs survive the limit check.
        let (ppd, _) = vertical_tmp_grad_ppd(
            vtg_inputs(25.0, 25.0, 0.1, 1.2, 0.5, 2.0),
            Default::default(),
        );
        assert!(
            !ppd.is_nan(),
            "in-range inputs should not be filtered by limit_inputs=true"
        );
    }

    /// `round_output` is upstream's, and was hardcoded to `true` here. Turning it off
    /// must expose digits that rounding to one decimal place would have removed.
    #[test]
    fn vertical_tmp_grad_ppd_round_output_can_be_turned_off() {
        let rounded = vertical_tmp_grad_ppd(
            vtg_inputs(25.0, 25.0, 0.1, 1.2, 0.5, 3.7),
            Default::default(),
        );
        let exact = vertical_tmp_grad_ppd(
            vtg_inputs(25.0, 25.0, 0.1, 1.2, 0.5, 3.7),
            VerticalTmpGradPpdOptions {
                round_output: false,
                limit_inputs: true,
            },
        );

        let (r, _) = rounded;
        let (e, _) = exact;
        assert!(
            (r * 10.0 - (r * 10.0).round()).abs() < 1e-9,
            "rounded ppd {r} is not at one decimal"
        );
        assert!(
            (e - r).abs() > 1e-12,
            "unrounded ppd {e} equals the rounded {r}"
        );
        assert!(
            (e - r).abs() < 0.05,
            "unrounded ppd {e} is not within rounding of {r}"
        );
    }
}
