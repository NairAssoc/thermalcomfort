//! Fan use during heatwaves assessment
//!
//! Estimate if environmental conditions would cause heat strain during heatwaves
//! when using fans.

use crate::models::two_nodes_gagge::{GaggeTwoNodesOptions, two_nodes_gagge};
use crate::utilities::Posture;
use crate::{ClothingInsulation, MetabolicRate};
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// Result of fan use during heatwaves assessment
#[derive(Debug, Clone, Copy)]
pub struct UseFansHeatwavesResult {
    /// Total evaporative heat loss from skin [W/m²]
    pub e_skin: f64,
    /// Heat lost by evaporation of regulatory sweat [W/m²]
    pub e_rsw: f64,
    /// Maximum evaporative capacity [W/m²]
    pub e_max: f64,
    /// Sensible heat loss [W/m²]
    pub q_sensible: f64,
    /// Total heat loss from skin [W/m²]
    pub q_skin: f64,
    /// Heat loss by respiration [W/m²]
    pub q_res: f64,
    /// Core temperature [°C]
    pub t_core: f64,
    /// Skin temperature [°C]
    pub t_skin: f64,
    /// Skin blood flow [kg/h/m²]
    pub m_bl: f64,
    /// Regulatory sweat generation [kg/h/m²]
    pub m_rsw: f64,
    /// Skin wettedness [0-1]
    pub w: f64,
    /// Maximum skin wettedness [0-1]
    pub w_max: f64,
    /// Heat strain from blood flow (m_bl at maximum)
    pub heat_strain_blood_flow: Option<bool>,
    /// Heat strain from wettedness (w at maximum)
    pub heat_strain_w: Option<bool>,
    /// Heat strain from sweating (m_rsw at maximum)
    pub heat_strain_sweating: Option<bool>,
    /// Overall heat strain indicator
    ///
    /// `None` when the inputs fell outside the applicability limits, matching Python's
    /// `nan` for this field. Reporting `false` there would assert "no heat strain" for
    /// a calculation the model declined to make.
    pub heat_strain: Option<bool>,
}

/// Estimate if conditions would cause heat strain during heatwaves
///
/// Determines whether the given environmental conditions would cause heat strain
/// when using fans. Heat strain occurs when any of the following reaches maximum:
/// - Regulatory sweat rate (m_rsw)
/// - Skin wettedness (w)
/// - Skin blood flow (m_bl)
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()`, recommended range: 20-50°C)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()`, recommended range: 20-50°C)
/// * `air_speed` - Air speed (use `Speed::from_meters_per_second()`, recommended range: 0.1-4.5 m/s)
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `metabolic_rate` - Metabolic rate (recommended range: 0.7-2.0 met)
/// * `clothing_insulation` - Clothing insulation (recommended range: 0-1 clo)
/// * `wme` - External work (default 0)
/// * `body_surface_area` - Body surface area (use `Area::from_square_meters()`, default 1.8258 m²)
/// * `p_atm` - Atmospheric pressure (use `Pressure::from_pascals()`, default 101325 Pa)
/// * `posture` - Body posture
/// * `max_skin_blood_flow` - Maximum blood flow [kg/h/m², default 80]
/// * `max_sweating` - Maximum sweat rate [kg/h/m², default 500]
///
/// # Returns
///
/// UseFansHeatwavesResult with physiological variables and heat strain indicators
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::use_fans_heatwaves::use_fans_heatwaves;
/// use thermalcomfort::utilities::Posture;
/// use thermalcomfort::{Temperature, Speed, Area, Pressure, Humidity, MetabolicRate, ClothingInsulation};
///
/// let result = use_fans_heatwaves(
///     Temperature::from_celsius(35.0),
///     Temperature::from_celsius(35.0),
///     Speed::from_meters_per_second(1.0),
///     Humidity::from_percent(50.0),
///     MetabolicRate::from_met(1.2),
///     ClothingInsulation::from_clo(0.5),
///     MetabolicRate::from_met(0.0),
///     Area::from_square_meters(1.8258),
///     Pressure::from_pascals(101325.0),
///     Posture::Standing,
///     80.0,
///     500.0,
///     true,  // limit_inputs
///     true,  // round_output
/// );
/// assert!(result.e_skin > 0.0);
/// assert_eq!(result.heat_strain, Some(false));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn use_fans_heatwaves(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    air_speed: Speed,
    relative_humidity: Humidity,
    metabolic_rate: MetabolicRate,
    clothing_insulation: ClothingInsulation,
    wme: MetabolicRate,
    body_surface_area: Area,
    p_atm: Pressure,
    posture: Posture,
    max_skin_blood_flow: f64,
    max_sweating: f64,
    limit_inputs: bool,
    round_output: bool,
) -> UseFansHeatwavesResult {
    // Run two-nodes Gagge model
    let options = GaggeTwoNodesOptions {
        wme,
        body_surface_area,
        p_atm,
        posture,
        max_skin_blood_flow,
        max_sweating,
        // Round once, here. Letting the Gagge model round first and rounding again
        // below is a double-rounding error: t_core 37.148833 became 37.15 then 37.2,
        // where pythermalcomfort reports 37.1. The heat-strain thresholds below also
        // need the unrounded values to compare meaningfully.
        round_output: false,
        ..Default::default()
    };

    let gagge_result = two_nodes_gagge(
        dry_bulb_temp,
        mean_radiant_temp,
        air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        options,
    );

    // Detect heat strain conditions.
    //
    // Exact equality, not a tolerance, matching pythermalcomfort. That is sound because
    // each quantity is *clamped* to its cap inside the two-node model, so a saturated
    // value is bit-identical to the cap. A tolerance instead reports strain for values
    // merely near the cap: at tdb=38.1, tr=43.7, v=2.16 the previous 1e-3 window called
    // heat_strain_w true where Python reports false.
    #[allow(clippy::float_cmp)]
    let heat_strain_blood_flow = gagge_result.m_bl == max_skin_blood_flow;
    #[allow(clippy::float_cmp)]
    let heat_strain_w = gagge_result.w == gagge_result.w_max;
    #[allow(clippy::float_cmp)]
    let heat_strain_sweating = gagge_result.m_rsw == max_sweating;
    let heat_strain = heat_strain_blood_flow || heat_strain_w || heat_strain_sweating;

    // ASHRAE 55 / Jay et al. applicability limits. Outside them pythermalcomfort masks
    // every output to NaN, and the heat-strain verdicts to false, rather than reporting
    // a fan recommendation the model cannot stand behind.
    let within_limits = !limit_inputs
        || ((20.0..=50.0).contains(&dry_bulb_temp.as_celsius())
            && (20.0..=50.0).contains(&mean_radiant_temp.as_celsius())
            && (0.1..=4.5).contains(&air_speed.as_meters_per_second())
            && (0.7..=2.0).contains(&metabolic_rate.as_met())
            && (0.0..=1.0).contains(&clothing_insulation.as_clo()));

    if !within_limits {
        return UseFansHeatwavesResult {
            e_skin: f64::NAN,
            e_rsw: f64::NAN,
            e_max: f64::NAN,
            q_sensible: f64::NAN,
            q_skin: f64::NAN,
            q_res: f64::NAN,
            t_core: f64::NAN,
            t_skin: f64::NAN,
            m_bl: f64::NAN,
            m_rsw: f64::NAN,
            w: f64::NAN,
            w_max: f64::NAN,
            heat_strain_blood_flow: None,
            heat_strain_w: None,
            heat_strain_sweating: None,
            heat_strain: None,
        };
    }

    let round1 = |x: f64| {
        if round_output {
            crate::utilities::round_half_even(x * 10.0) / 10.0
        } else {
            x
        }
    };

    UseFansHeatwavesResult {
        e_skin: round1(gagge_result.e_skin),
        e_rsw: round1(gagge_result.e_rsw),
        e_max: round1(gagge_result.e_max),
        q_sensible: round1(gagge_result.q_sensible),
        q_skin: round1(gagge_result.q_skin),
        q_res: round1(gagge_result.q_res),
        t_core: round1(gagge_result.t_core),
        t_skin: round1(gagge_result.t_skin),
        m_bl: round1(gagge_result.m_bl),
        m_rsw: round1(gagge_result.m_rsw),
        w: round1(gagge_result.w),
        w_max: round1(gagge_result.w_max),
        heat_strain_blood_flow: Some(heat_strain_blood_flow),
        heat_strain_w: Some(heat_strain_w),
        heat_strain_sweating: Some(heat_strain_sweating),
        heat_strain: Some(heat_strain),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_use_fans_heatwaves() {
        let result = use_fans_heatwaves(
            Temperature::from_celsius(35.0),
            Temperature::from_celsius(35.0),
            Speed::from_meters_per_second(1.0),
            Humidity::from_percent(50.0),
            MetabolicRate::from_met(1.2),
            ClothingInsulation::from_clo(0.5),
            MetabolicRate::from_met(0.0),
            Area::from_square_meters(1.8258),
            Pressure::from_pascals(101325.0),
            Posture::Standing,
            80.0,
            500.0,
            true,
            true,
        );
        assert!(result.e_skin > 0.0);
        assert!(result.t_core > 36.0 && result.t_core < 39.0);
    }

    #[test]
    fn test_heat_strain_detection() {
        // Extreme conditions that should trigger heat strain
        let result = use_fans_heatwaves(
            Temperature::from_celsius(45.0),
            Temperature::from_celsius(45.0),
            Speed::from_meters_per_second(0.5),
            Humidity::from_percent(70.0),
            MetabolicRate::from_met(1.8),
            ClothingInsulation::from_clo(0.3),
            MetabolicRate::from_met(0.0),
            Area::from_square_meters(1.8258),
            Pressure::from_pascals(101325.0),
            Posture::Standing,
            80.0,
            500.0,
            true,
            true,
        );
        // Should detect some form of heat strain in extreme conditions
        assert!(result.t_core > 37.0);
    }
}
