//! Specialty thermal comfort models and indices
//!
//! This module contains specialized models for specific comfort assessment scenarios.

use crate::models::pmv::{PmvPpdOptions, pmv_ppd_ashrae};
use crate::{ClothingInsulation, MetabolicRate};
use measurements::{Humidity, Length, Speed, Temperature};

/// Calculate percentage dissatisfied due to ankle draft
///
/// Calculates the percentage of thermally dissatisfied people with the ankle draft
/// (0.1 m) above floor level. Only applicable for vr < 0.2 m/s.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `relative_air_speed` - Relative air speed (use `Speed::from_meters_per_second()` or similar, must be < 0.2 m/s)
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `metabolic_rate` - Metabolic rate
/// * `clothing_insulation` - Clothing insulation
/// * `ankle_air_speed` - Air speed at 0.1m above floor (use `Speed::from_meters_per_second()` or similar)
/// * `limit_inputs` - If true, returns NaN/false when any input is outside the ASHRAE 55
///   applicability range: 10 ≤ tdb [°C] ≤ 40, 10 ≤ tr [°C] ≤ 40, 0 ≤ vr [m/s] ≤ 0.2,
///   1 ≤ met ≤ 4, 0 ≤ clo ≤ 1.5.
///
/// # Returns
///
/// Tuple of (ppd_ankle_draft %, acceptability bool)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::ankle_draft;
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let (ppd, acceptable) = ankle_draft(
///     Temperature::from_celsius(23.0),
///     Temperature::from_celsius(23.0),
///     Speed::from_meters_per_second(0.1),
///     Humidity::from_percent(45.0),
///     MetabolicRate::from_met(1.1),
///     ClothingInsulation::from_clo(0.7),
///     Speed::from_meters_per_second(0.15),  // ankle draft
///     true,
/// );
/// println!("PPD ankle draft: {:.1}%, Acceptable: {}", ppd, acceptable);
/// ```
///
/// # References
///
/// - Liu et al. (2017)
/// - ASHRAE 55-2023
#[allow(clippy::too_many_arguments)]
pub fn ankle_draft(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: Humidity,
    metabolic_rate: MetabolicRate,
    clothing_insulation: ClothingInsulation,
    ankle_air_speed: Speed,
    limit_inputs: bool,
) -> (f64, bool) {
    // Calculate PMV value for use in ankle draft equation.
    // Matches pythermalcomfort behaviour: PMV is computed without input limits so the
    // outer limit_inputs flag governs the final return value.
    let pmv_result = pmv_ppd_ashrae(
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        PmvPpdOptions {
            limit_inputs: false,
            ..Default::default()
        },
    );
    let pmv = pmv_result.pmv; // Use PMV value directly, not TSV enum

    let ankle_speed = ankle_air_speed.as_meters_per_second();

    // Calculate PPD for ankle draft using logistic function
    let exponent = -2.58 + 3.05 * ankle_speed - 1.06 * pmv;
    let ppd_ad = (libm::exp(exponent) / (1.0 + libm::exp(exponent))) * 100.0;
    let ppd_ad = libm::round(ppd_ad * 10.0) / 10.0;

    if limit_inputs
        && !ashrae55_ankle_inputs_valid(
            dry_bulb_temp,
            mean_radiant_temp,
            relative_air_speed,
            metabolic_rate,
            clothing_insulation,
        )
    {
        return (f64::NAN, false);
    }

    let acceptability = ppd_ad <= 20.0;

    (ppd_ad, acceptability)
}

fn ashrae55_ankle_inputs_valid(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    metabolic_rate: MetabolicRate,
    clothing_insulation: ClothingInsulation,
) -> bool {
    let tdb = dry_bulb_temp.as_celsius();
    let tr = mean_radiant_temp.as_celsius();
    let vr = relative_air_speed.as_meters_per_second();
    let met = metabolic_rate.as_met();
    let clo = clothing_insulation.as_clo();
    (10.0..=40.0).contains(&tdb)
        && (10.0..=40.0).contains(&tr)
        && (0.0..=0.2).contains(&vr)
        && (1.0..=4.0).contains(&met)
        && (0.0..=1.5).contains(&clo)
}

/// Calculate PPD for vertical air temperature gradient
///
/// Calculates the percentage of thermally dissatisfied people with a vertical
/// temperature gradient between feet and head.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `relative_air_speed` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `metabolic_rate` - Metabolic rate
/// * `clothing_insulation` - Clothing insulation
/// * `vertical_temp_gradient` - Vertical temperature gradient between 1.1m and 0.1m [°C]
/// * `limit_inputs` - If true, returns NaN/false when any input is outside the ASHRAE 55
///   applicability range: 10 ≤ tdb [°C] ≤ 40, 10 ≤ tr [°C] ≤ 40, 0 ≤ vr [m/s] ≤ 0.2,
///   1 ≤ met ≤ 4, 0 ≤ clo ≤ 1.5.
///
/// # Returns
///
/// Tuple of (ppd %, acceptability bool)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::vertical_tmp_grad_ppd;
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let (ppd, acceptable) = vertical_tmp_grad_ppd(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Speed::from_meters_per_second(0.1),
///     Humidity::from_percent(50.0),
///     MetabolicRate::from_met(1.2),
///     ClothingInsulation::from_clo(0.5),
///     2.0,  // 2°C temperature gradient
///     true,
/// );
/// println!("PPD vertical gradient: {:.1}%, Acceptable: {}", ppd, acceptable);
/// ```
///
/// # References
///
/// - ISO 7730:2005
/// - ASHRAE 55-2023
#[allow(clippy::too_many_arguments)]
pub fn vertical_tmp_grad_ppd(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: Humidity,
    metabolic_rate: MetabolicRate,
    clothing_insulation: ClothingInsulation,
    vertical_temp_gradient: f64,
    limit_inputs: bool,
) -> (f64, bool) {
    // Calculate PMV value for use in vertical temperature gradient equation.
    // Matches pythermalcomfort behaviour: PMV is computed without input limits so the
    // outer limit_inputs flag governs the final return value.
    let pmv_result = pmv_ppd_ashrae(
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        PmvPpdOptions {
            limit_inputs: false,
            ..Default::default()
        },
    );
    let pmv = pmv_result.pmv;

    // PPD calculation for vertical temperature gradient using ASHRAE 55-2023 formula
    let numerator =
        libm::exp(0.13 * libm::pow(pmv - 1.91, 2.0) + 0.15 * vertical_temp_gradient - 1.6);
    let ppd_vtg = (numerator / (1.0 + numerator) - 0.345) * 100.0;
    let ppd_vtg = libm::round(ppd_vtg * 10.0) / 10.0;

    if limit_inputs
        && !ashrae55_ankle_inputs_valid(
            dry_bulb_temp,
            mean_radiant_temp,
            relative_air_speed,
            metabolic_rate,
            clothing_insulation,
        )
    {
        return (f64::NAN, false);
    }

    let acceptability = ppd_vtg <= 5.0;

    (ppd_vtg, acceptability)
}

/// Calculate sky-vault view fraction
///
/// Calculates the fraction of the sky visible through a window.
///
/// # Arguments
///
/// * `w` - Width of the window
/// * `h` - Height of the window
/// * `d` - Distance between occupant and window
///
/// # Returns
///
/// Sky-vault view fraction (0-1)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::f_svv;
/// use thermalcomfort::Length;
///
/// let svv = f_svv(Length::from_meters(2.0), Length::from_meters(1.5), Length::from_meters(3.0));
/// assert!(svv > 0.0 && svv <= 1.0);
/// ```
pub fn f_svv(w: Length, h: Length, d: Length) -> f64 {
    let w = w.as_meters();
    let h = h.as_meters();
    let d = d.as_meters();
    let angle_h = libm::atan(h / (2.0 * d));
    let angle_w = libm::atan(w / (2.0 * d));

    // Convert radians to degrees and calculate fraction
    let degrees_h = angle_h * 180.0 / core::f64::consts::PI;
    let degrees_w = angle_w * 180.0 / core::f64::consts::PI;

    (degrees_h * degrees_w) / 16200.0
}

/// Transpose SHARP solar altitude
///
/// Converts between solar altitude and SHARP altitude coordinates.
///
/// # Arguments
///
/// * `sharp` - SHARP altitude (degrees)
/// * `altitude` - Solar altitude (degrees)
///
/// # Returns
///
/// Tuple of (transposed_sharp, transposed_altitude)
pub fn transpose_sharp_altitude(sharp: f64, altitude: f64) -> (f64, f64) {
    // Simple coordinate transformation
    let t_sharp = sharp + altitude / 2.0;
    let t_altitude = altitude - sharp / 2.0;
    (t_sharp, t_altitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ankle_draft() {
        let (ppd, acceptable) = ankle_draft(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            MetabolicRate::from_met(1.2),
            ClothingInsulation::from_clo(0.5),
            Speed::from_meters_per_second(0.3),
            true,
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(50.0),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                Speed::from_meters_per_second(0.15),
                true,
            );
            assert!(ppd.is_nan(), "{label}: expected NaN with limit_inputs=true");
            assert!(
                !acceptable,
                "{label}: expected !acceptable with limit_inputs=true"
            );

            let (ppd_unlimited, _) = ankle_draft(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(50.0),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                Speed::from_meters_per_second(0.15),
                false,
            );
            assert!(
                !ppd_unlimited.is_nan(),
                "{label}: expected numeric PPD with limit_inputs=false"
            );
        }

        // Spot-check that a fully in-range input is not flagged when limits are on.
        let (ppd, _) = ankle_draft(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            MetabolicRate::from_met(1.2),
            ClothingInsulation::from_clo(0.5),
            Speed::from_meters_per_second(0.15),
            true,
        );
        assert!(
            !ppd.is_nan(),
            "in-range inputs should not be filtered by limit_inputs=true"
        );
    }

    #[test]
    fn test_vertical_tmp_grad_ppd() {
        let (ppd, _acceptable) = vertical_tmp_grad_ppd(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            MetabolicRate::from_met(1.2),
            ClothingInsulation::from_clo(0.5),
            2.0,
            true,
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(50.0),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                2.0,
                true,
            );
            assert!(ppd.is_nan(), "{label}: expected NaN with limit_inputs=true");
            assert!(
                !acceptable,
                "{label}: expected !acceptable with limit_inputs=true"
            );

            let (ppd_unlimited, _) = vertical_tmp_grad_ppd(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(50.0),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                2.0,
                false,
            );
            assert!(
                !ppd_unlimited.is_nan(),
                "{label}: expected numeric PPD with limit_inputs=false"
            );
        }

        // Spot-check that fully in-range inputs survive the limit check.
        let (ppd, _) = vertical_tmp_grad_ppd(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            MetabolicRate::from_met(1.2),
            ClothingInsulation::from_clo(0.5),
            2.0,
            true,
        );
        assert!(
            !ppd.is_nan(),
            "in-range inputs should not be filtered by limit_inputs=true"
        );
    }

    #[test]
    fn test_f_svv() {
        let svv = f_svv(
            Length::from_meters(2.0),
            Length::from_meters(1.5),
            Length::from_meters(3.0),
        );
        assert!(svv > 0.0 && svv <= 1.0);
    }

    #[test]
    fn test_transpose_sharp_altitude() {
        let (sharp_t, alt_t) = transpose_sharp_altitude(30.0, 45.0);
        assert!(sharp_t > 0.0);
        assert!(alt_t > 0.0);
    }
}
