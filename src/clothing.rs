//! Clothing insulation: area factor, dynamic insulation under ASHRAE 55 and ISO 9920,
//! air-layer and total insulation, environmental correction, and ensemble summation.
//! Mirrors `pythermalcomfort.clothing`.

use crate::constants::MET_TO_W_M2;
use crate::environment::v_relative;
use crate::utilities::round_to;
use crate::{ClothingInsulation, MetabolicRate};
use libm::{exp, pow};
use measurements::Speed;

/// Calculate clothing area factor
///
/// # Arguments
///
/// * `i_cl` - Intrinsic clothing insulation (clo)
///
/// # Returns
///
/// Clothing area factor
///
/// # Source
/// ISO 9920: f_cl = 1.0 + 0.28 * i_cl
/// where 0.28 is the empirical coefficient relating clothing insulation to increased surface area
#[inline]
pub fn clo_area_factor(i_cl: ClothingInsulation) -> f64 {
    1.0 + 0.28 * i_cl.as_clo()
}

/// Edition selector for ASHRAE 55-derived calculations.
///
/// pythermalcomfort's `clo_dynamic_ashrae` and `pmv_ppd_ashrae` both take a `model: str`
/// parameter (default `"55-2023"`) that is validated against a single legal value and
/// never branched on — today there is exactly one ASHRAE 55 edition these functions
/// support. The selector is still ported (rather than dropped) because it is a
/// documented, user-facing part of upstream's signature; a future edition would add a
/// variant here, and every match on this enum is exhaustive so that addition is a
/// compile error everywhere it needs handling, not a silent no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ashrae55Model {
    /// ASHRAE 55-2023 (the only edition currently supported by either function).
    #[default]
    Ashrae552023,
}

/// Edition selector for [`clo_dynamic_iso`].
///
/// Mirrors [`Ashrae55Model`]: pythermalcomfort's `clo_dynamic_iso` takes a `model: str`
/// parameter (default `"9920-2007"`) validated against a single legal value. Ported as a
/// single-variant enum for the same reason — it is part of upstream's public signature,
/// and a future edition becomes a compile error to ignore rather than a silent no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Iso9920Model {
    /// ISO 9920:2007 (the only edition currently supported).
    #[default]
    Iso99202007,
}

/// The inputs to [`clo_dynamic_ashrae`]: both parameters are required in
/// pythermalcomfort's signature (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloDynamicAshraeInputs {
    /// Static (intrinsic) clothing insulation
    pub clo: ClothingInsulation,
    /// Metabolic rate
    pub met: MetabolicRate,
}

/// Optional parameters for [`clo_dynamic_ashrae`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CloDynamicAshraeOptions {
    /// Standard edition. Behaviourally identical to the only other legal value today
    /// (there is only one), see [`Ashrae55Model`].
    pub model: Ashrae55Model,
}

/// Calculate dynamic clothing insulation for ASHRAE 55
///
/// # Returns
///
/// Dynamic clothing insulation
///
/// # Source
/// ASHRAE 55: For met > 1.2, clo_dyn = clo * (0.6 + 0.4/met)
/// - 1.2 met: threshold for walking/active movement
/// - 0.6: base reduction factor
/// - 0.4: adjustment factor (accounts for increased ventilation with activity)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{clo_dynamic_ashrae, CloDynamicAshraeInputs, CloDynamicAshraeOptions};
/// use thermalcomfort::{ClothingInsulation, MetabolicRate};
///
/// let clo_dyn = clo_dynamic_ashrae(
///     CloDynamicAshraeInputs {
///         clo: ClothingInsulation::from_clo(0.5),
///         met: MetabolicRate::from_met(1.4),
///     },
///     CloDynamicAshraeOptions::default(),
/// );
/// assert!(clo_dyn.as_clo() < 0.5);
/// ```
#[inline]
pub fn clo_dynamic_ashrae(
    inputs: CloDynamicAshraeInputs,
    options: CloDynamicAshraeOptions,
) -> ClothingInsulation {
    let CloDynamicAshraeInputs { clo, met } = inputs;
    // Exhaustive match: today there is only one edition, but this stops a future
    // variant from being silently ignored.
    match options.model {
        Ashrae55Model::Ashrae552023 => {}
    }
    let met_val = met.as_met();
    if met_val > 1.2 {
        ClothingInsulation::from_clo(round_to(clo.as_clo() * (0.6 + 0.4 / met_val), 3))
    } else {
        clo
    }
}

/// Inputs for [`clo_insulation_air_layer`].
///
/// Named rather than positional because `vr` and `v_walk` are both [`Speed`] and mean
/// different things — relative air speed against the occupant's walking speed — so a
/// transposed pair is a wrong answer rather than a compile error. Field names are
/// upstream's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloInsulationAirLayerInputs {
    /// Relative air speed
    pub vr: Speed,
    /// Walking speed
    pub v_walk: Speed,
    /// Static boundary air layer insulation (typically 0.7 clo)
    pub i_a_static: ClothingInsulation,
}

/// Inputs for [`clo_total_insulation`].
///
/// Five values over two types — three [`ClothingInsulation`] and two [`Speed`] — each group
/// silently interchangeable while positional. Field names are upstream's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloTotalInsulationInputs {
    /// Static total insulation of the clothing ensemble
    pub i_t: ClothingInsulation,
    /// Relative air speed
    pub vr: Speed,
    /// Walking speed
    pub v_walk: Speed,
    /// Static boundary air layer insulation
    pub i_a_static: ClothingInsulation,
    /// Static basic insulation of the clothing
    pub i_cl: ClothingInsulation,
}

/// Inputs for [`clo_correction_factor_environment`].
///
/// `vr` and `v_walk` are both [`Speed`]; see [`CloInsulationAirLayerInputs`]. Field names
/// are upstream's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloCorrectionFactorEnvironmentInputs {
    /// Relative air speed
    pub vr: Speed,
    /// Walking speed
    pub v_walk: Speed,
    /// Static basic insulation of the clothing
    pub i_cl: ClothingInsulation,
}

/// Calculate insulation of the boundary air layer (I_a,r) - ISO 9920:2007
///
/// The static boundary air value is 0.7 clo for air velocities around 0.1-0.15 m/s.
/// For walking conditions, the boundary air layer insulation is calculated based on
/// the walking speed and relative air speed.
///
/// # Arguments
///
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `v_walk` - Walking speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_a_static` - Static boundary air layer insulation (typically 0.7 clo)
///
/// # Returns
///
/// Boundary air layer insulation (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{clo_insulation_air_layer, CloInsulationAirLayerInputs};
/// use thermalcomfort::{Speed, ClothingInsulation};
///
/// let i_a_r = clo_insulation_air_layer(CloInsulationAirLayerInputs {
///     vr: Speed::from_meters_per_second(0.1),
///     v_walk: Speed::from_meters_per_second(0.0),
///     i_a_static: ClothingInsulation::from_clo(0.7),
/// });
/// assert!((i_a_r.as_clo() - 0.719).abs() < 0.01);
/// ```
#[inline]
pub fn clo_insulation_air_layer(inputs: CloInsulationAirLayerInputs) -> ClothingInsulation {
    let CloInsulationAirLayerInputs {
        vr,
        v_walk,
        i_a_static,
    } = inputs;
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    ClothingInsulation::from_clo(
        exp(
            -0.533 * (vr_ms - 0.15) + 0.069 * pow(vr_ms - 0.15, 2.0) - 0.462 * v_walk_ms
                + 0.201 * pow(v_walk_ms, 2.0),
        ) * i_a_static.as_clo(),
    )
}

/// Correction factor for nude person - ISO 9920:2007
///
/// Empirical coefficients from ISO 9920:
/// - -0.533: linear velocity coefficient
/// - 0.15 m/s: reference air speed offset
/// - 0.069: quadratic velocity coefficient
/// - -0.462: linear walking speed coefficient
/// - 0.201: quadratic walking speed coefficient
#[inline]
fn correction_nude(vr: Speed, v_walk: Speed) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.533 * (vr_ms - 0.15) + 0.069 * pow(vr_ms - 0.15, 2.0) - 0.462 * v_walk_ms
            + 0.201 * pow(v_walk_ms, 2.0),
    )
}

/// Correction factor for normal clothing - ISO 9920:2007
///
/// Empirical coefficients from ISO 9920 for clothed persons:
/// - -0.281: linear velocity coefficient
/// - 0.15 m/s: reference air speed offset
/// - 0.044: quadratic velocity coefficient
/// - -0.492: linear walking speed coefficient
#[inline]
fn correction_normal_clothing(vr: Speed, v_walk: Speed) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.281 * (vr_ms - 0.15) + 0.044 * pow(vr_ms - 0.15, 2.0) - 0.492 * v_walk_ms
            + 0.176 * pow(v_walk_ms, 2.0),
    )
}

/// Calculate total insulation of clothing ensemble (I_T,r) - ISO 9920:2007
///
/// The total insulation (I_T,r) is the actual thermal insulation from the body surface
/// to the environment, considering all clothing, enclosed air layers, and boundary air
/// layers under given environmental conditions and activities.
///
/// Different equations are used based on clothing level:
/// - Nude: i_cl = 0 clo
/// - Low clothing: i_cl < 0.6 clo
/// - Normal clothing: 0.6 clo <= i_cl <= 1.4 clo
///
/// # Arguments
///
/// * `i_t` - Total thermal insulation under static conditions
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `v_walk` - Walking speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_a_static` - Static boundary air layer insulation
/// * `i_cl` - Intrinsic clothing insulation
///
/// # Returns
///
/// Total insulation of clothing ensemble (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{clo_total_insulation, CloTotalInsulationInputs};
/// use thermalcomfort::{Speed, ClothingInsulation};
///
/// let i_t_r = clo_total_insulation(CloTotalInsulationInputs {
///     i_t: ClothingInsulation::from_clo(1.5),
///     vr: Speed::from_meters_per_second(0.1),
///     v_walk: Speed::from_meters_per_second(0.0),
///     i_a_static: ClothingInsulation::from_clo(0.7),
///     i_cl: ClothingInsulation::from_clo(1.0),
/// });
/// assert!(i_t_r.as_clo() > 0.0);
/// ```
pub fn clo_total_insulation(inputs: CloTotalInsulationInputs) -> ClothingInsulation {
    let CloTotalInsulationInputs {
        i_t,
        vr,
        v_walk,
        i_a_static,
        i_cl,
    } = inputs;
    let i_t = i_t.as_clo();
    let i_a_static = i_a_static.as_clo();
    let i_cl = i_cl.as_clo();
    // Calculate insulation for different clothing levels
    let nude_insulation = i_a_static * correction_nude(vr, v_walk);
    let normal_insulation = i_t * correction_normal_clothing(vr, v_walk);

    let i_t_r = if i_cl == 0.0 {
        // Nude
        nude_insulation
    } else if i_cl <= 0.6 {
        // Low clothing - interpolate between nude and normal
        ((0.6 - i_cl) * nude_insulation + i_cl * normal_insulation) / 0.6
    } else {
        // Normal clothing
        normal_insulation
    };
    ClothingInsulation::from_clo(i_t_r)
}

/// The inputs to [`clo_dynamic_iso`]: all three are required in pythermalcomfort's
/// signature (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloDynamicIsoInputs {
    /// Static (intrinsic) clothing insulation
    pub clo: ClothingInsulation,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Air speed (not yet the relative air speed; computed internally via
    /// [`v_relative`])
    pub v: Speed,
}

/// Optional parameters for [`clo_dynamic_iso`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloDynamicIsoOptions {
    /// Thermal insulation of the boundary (surface) air layer around the outer
    /// clothing or, when nude, around the skin surface. Defaults to 0.7 clo.
    pub i_a: ClothingInsulation,
    /// Standard edition. Behaviourally identical to the only other legal value today
    /// (there is only one), see [`Iso9920Model`].
    pub model: Iso9920Model,
}

impl Default for CloDynamicIsoOptions {
    fn default() -> Self {
        Self {
            i_a: ClothingInsulation::from_clo(0.7),
            model: Iso9920Model::default(),
        }
    }
}

/// Calculate dynamic clothing insulation for ISO 9920:2007
///
/// Estimates the dynamic intrinsic clothing insulation (I_cl,r). The activity
/// as well as the air speed modify the insulation characteristics of the clothing.
///
/// # Returns
///
/// Dynamic clothing insulation (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{clo_dynamic_iso, CloDynamicIsoInputs, CloDynamicIsoOptions};
/// use thermalcomfort::{Speed, ClothingInsulation, MetabolicRate};
///
/// let clo_dyn = clo_dynamic_iso(
///     CloDynamicIsoInputs {
///         clo: ClothingInsulation::from_clo(1.0),
///         met: MetabolicRate::from_met(1.2),
///         v: Speed::from_meters_per_second(0.1),
///     },
///     CloDynamicIsoOptions::default(),
/// );
/// assert!(clo_dyn.as_clo() > 0.0 && clo_dyn.as_clo() <= 1.0);
/// ```
pub fn clo_dynamic_iso(
    inputs: CloDynamicIsoInputs,
    options: CloDynamicIsoOptions,
) -> ClothingInsulation {
    let CloDynamicIsoInputs { clo, met, v } = inputs;
    let CloDynamicIsoOptions { i_a, model } = options;
    // Exhaustive match: today there is only one edition, but this stops a future
    // variant from being silently ignored.
    match model {
        Iso9920Model::Iso99202007 => {}
    }

    let clo_val = clo.as_clo();
    // Calculate clothing area factor
    let f_cl = clo_area_factor(clo);

    // Total insulation under static conditions
    let i_t = clo_val + i_a.as_clo() / f_cl;

    // Relative air speed for the whole-body heat balance
    let v_r = v_relative(v, met);

    // Walking speed when undefined, per ISO 7730 Annex C / ISO 9920:
    // v_walk = 0.0052 * (M - 58), clipped to [0, 0.7] m/s, where M is metabolic rate
    // in W/m². This is a distinct formula from `v_relative`'s activity-generated air
    // speed, which the previous implementation incorrectly reused here.
    let v_walk_ms = (0.0052 * (met.as_met() * MET_TO_W_M2 - 58.0)).clamp(0.0, 0.7);
    let v_walk = Speed::from_meters_per_second(v_walk_ms);

    // Calculate total dynamic insulation
    let i_t_r = clo_total_insulation(CloTotalInsulationInputs {
        i_t: ClothingInsulation::from_clo(i_t),
        vr: v_r,
        v_walk,
        i_a_static: i_a,
        i_cl: clo,
    });

    // Calculate dynamic air layer insulation
    let i_a_r = clo_insulation_air_layer(CloInsulationAirLayerInputs {
        vr: v_r,
        v_walk,
        i_a_static: i_a,
    });

    // Return dynamic clothing insulation
    ClothingInsulation::from_clo(i_t_r.as_clo() - i_a_r.as_clo() / f_cl)
}
/// Calculate environmental correction factor for clothing insulation
///
/// Returns the correction factor for total clothing insulation to account for
/// real environmental conditions (walking, air movement, etc.) vs. static
/// measurement conditions. Based on ISO 9920:2007.
///
/// # Arguments
///
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `v_walk` - Walking speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_cl` - Intrinsic clothing insulation
///
/// # Returns
///
/// Correction factor for clothing insulation
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{
///     clo_correction_factor_environment, CloCorrectionFactorEnvironmentInputs,
/// };
/// use thermalcomfort::{Speed, ClothingInsulation};
///
/// let cf = clo_correction_factor_environment(CloCorrectionFactorEnvironmentInputs {
///     vr: Speed::from_meters_per_second(0.3),
///     v_walk: Speed::from_meters_per_second(0.5),
///     i_cl: ClothingInsulation::from_clo(0.8),
/// });
/// assert!(cf > 0.0 && cf <= 1.0);
/// ```
///
/// # References
///
/// - ISO 9920:2007
pub fn clo_correction_factor_environment(inputs: CloCorrectionFactorEnvironmentInputs) -> f64 {
    let CloCorrectionFactorEnvironmentInputs { vr, v_walk, i_cl } = inputs;
    let i_cl = i_cl.as_clo();
    if i_cl == 0.0 {
        return correction_nude(vr, v_walk);
    }

    if i_cl <= 0.6 {
        let nude_corr = correction_nude(vr, v_walk);
        let normal_corr = correction_normal_clothing(vr, v_walk);
        ((0.6 - i_cl) * nude_corr + i_cl * normal_corr) / 0.6
    } else {
        correction_normal_clothing(vr, v_walk)
    }
}

/// Calculate intrinsic insulation of clothing ensemble
///
/// Calculates the intrinsic insulation of a clothing ensemble based on the
/// sum of individual garment insulation values. Valid for ensembles with
/// rather uniform insulation distribution across the body.
///
/// # Arguments
///
/// * `clo_garments` - Slice of clothing insulation values for each garment (clo)
///
/// # Returns
///
/// Total intrinsic insulation of ensemble (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::clothing::clo_intrinsic_insulation_ensemble;
/// use thermalcomfort::ClothingInsulation;
///
/// let garments = [
///     ClothingInsulation::from_clo(0.25), // shirt
///     ClothingInsulation::from_clo(0.15), // pants
///     ClothingInsulation::from_clo(0.10), // underwear
/// ];
/// let total = clo_intrinsic_insulation_ensemble(&garments);
/// assert!(total.as_clo() > 0.0);
/// ```
///
/// # References
///
/// - ISO 9920:2009 Section 4.3
pub fn clo_intrinsic_insulation_ensemble(
    clo_garments: &[ClothingInsulation],
) -> ClothingInsulation {
    let sum: f64 = clo_garments.iter().map(|c| c.as_clo()).sum();
    ClothingInsulation::from_clo(sum * 0.835 + 0.161)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clo_dynamic_iso_matches_python() {
        // Reference values from pythermalcomfort 4.4.0 clo_dynamic_iso (i_a defaults to 0.7).
        // These exercise the ISO 7730 Annex C / ISO 9920 walking-speed formula.
        for (clo, met, v, expected) in [
            (0.5, 1.2, 0.1, 0.417585),
            (1.0, 2.0, 0.3, 0.817727),
            (0.7, 1.5, 0.2, 0.640019),
            (1.5, 3.0, 0.5, 1.000017),
        ] {
            let got = clo_dynamic_iso(
                CloDynamicIsoInputs {
                    clo: ClothingInsulation::from_clo(clo),
                    met: MetabolicRate::from_met(met),
                    v: Speed::from_meters_per_second(v),
                },
                CloDynamicIsoOptions::default(),
            );
            let got = got.as_clo();
            assert!(
                (got - expected).abs() < 1e-5,
                "clo_dynamic_iso({clo}, {met}, {v}) = {got}, expected {expected}"
            );
        }
    }
    #[test]
    fn test_clo_area_factor() {
        assert!((clo_area_factor(ClothingInsulation::from_clo(0.5)) - 1.14).abs() < 0.01);
        assert!((clo_area_factor(ClothingInsulation::from_clo(1.0)) - 1.28).abs() < 0.01);
    }
    #[test]
    fn test_clo_intrinsic_insulation_ensemble() {
        // Test with typical garments - shirt + pants + underwear
        let garments = [
            ClothingInsulation::from_clo(0.25), // long-sleeve shirt
            ClothingInsulation::from_clo(0.24), // thick trousers
            ClothingInsulation::from_clo(0.04), // underwear
        ];
        let total = clo_intrinsic_insulation_ensemble(&garments);
        // Formula: sum * 0.835 + 0.161 = 0.53 * 0.835 + 0.161 = 0.604
        assert!((total.as_clo() - 0.604).abs() < 0.01);

        // Test with single garment
        let single = [ClothingInsulation::from_clo(0.5)];
        let total_single = clo_intrinsic_insulation_ensemble(&single);
        // Formula: 0.5 * 0.835 + 0.161 = 0.579
        assert!((total_single.as_clo() - 0.579).abs() < 0.01);
    }
}
