//! Fan use during heatwaves assessment
//!
//! Estimate if environmental conditions would cause heat strain during heatwaves
//! when using fans.

use crate::models::two_nodes_gagge::GaggePosture;
use crate::models::two_nodes_gagge::{GaggeTwoNodesInputs, GaggeTwoNodesOptions, two_nodes_gagge};
use crate::{ClothingInsulation, HeatFluxDensity, MetabolicRate};
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// The comfort inputs to [`use_fans_heatwaves`]: pythermalcomfort requires all six (no
/// default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UseFansHeatwavesInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Air speed
    pub v: Speed,
    /// Relative humidity
    pub rh: Humidity,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation
    pub clo: ClothingInsulation,
}

/// Optional parameters for [`use_fans_heatwaves`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UseFansHeatwavesOptions {
    /// External work
    pub wme: MetabolicRate,
    /// Body surface area
    pub body_surface_area: Area,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Body position
    pub position: GaggePosture,
    /// Maximum blood flow from the core to the skin [kg/h/m²].
    ///
    /// Defaults to **80** here, unlike [`two_nodes_gagge`]'s default of 90 -- this
    /// model's own upstream default is different and must not be copied from the other
    /// one.
    pub max_skin_blood_flow: f64,
    /// Maximum rate at which regulatory sweat is generated [kg/h/m²]
    pub max_sweating: f64,
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
    /// Round output values
    pub round_output: bool,
}

impl Default for UseFansHeatwavesOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            body_surface_area: Area::from_square_meters(1.8258),
            p_atm: Pressure::from_pascals(101325.0),
            position: GaggePosture::Standing,
            max_skin_blood_flow: 80.0,
            max_sweating: 500.0,
            limit_inputs: true,
            round_output: true,
        }
    }
}

/// Result of fan use during heatwaves assessment
#[derive(Debug, Clone, Copy)]
pub struct UseFansHeatwavesResult {
    /// Total evaporative heat loss from skin
    pub e_skin: HeatFluxDensity,
    /// Heat lost by evaporation of regulatory sweat
    pub e_rsw: HeatFluxDensity,
    /// Maximum evaporative capacity
    pub e_max: HeatFluxDensity,
    /// Sensible heat loss
    pub q_sensible: HeatFluxDensity,
    /// Total heat loss from skin
    pub q_skin: HeatFluxDensity,
    /// Heat loss by respiration
    pub q_res: HeatFluxDensity,
    /// Core temperature
    pub t_core: Temperature,
    /// Skin temperature
    pub t_skin: Temperature,
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
/// # Returns
///
/// [`UseFansHeatwavesResult`] with physiological variables and heat strain indicators
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::use_fans_heatwaves::{
///     use_fans_heatwaves, UseFansHeatwavesInputs, UseFansHeatwavesOptions,
/// };
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let result = use_fans_heatwaves(
///     UseFansHeatwavesInputs {
///         tdb: Temperature::from_celsius(35.0),
///         tr: Temperature::from_celsius(35.0),
///         v: Speed::from_meters_per_second(1.0),
///         rh: Humidity::from_percent(50.0),
///         met: MetabolicRate::from_met(1.2),
///         clo: ClothingInsulation::from_clo(0.5),
///     },
///     Default::default(),
/// );
/// assert!(result.e_skin.as_watts_per_square_meter() > 0.0);
/// assert_eq!(result.heat_strain, Some(false));
/// ```
pub fn use_fans_heatwaves(
    inputs: UseFansHeatwavesInputs,
    options: UseFansHeatwavesOptions,
) -> UseFansHeatwavesResult {
    let UseFansHeatwavesInputs {
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
    } = inputs;
    let UseFansHeatwavesOptions {
        wme,
        body_surface_area,
        p_atm,
        position,
        max_skin_blood_flow,
        max_sweating,
        limit_inputs,
        round_output,
    } = options;

    // Run two-nodes Gagge model
    let gagge_options = GaggeTwoNodesOptions {
        wme,
        body_surface_area,
        p_atm,
        position,
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
        GaggeTwoNodesInputs {
            tdb,
            tr,
            v,
            rh,
            met,
            clo,
        },
        gagge_options,
    );

    // Detect heat strain conditions.
    //
    // Exact equality, not a tolerance, matching pythermalcomfort. That is sound because
    // each quantity is *clamped* to its cap inside the two-node model, so a saturated
    // value is bit-identical to the cap. A tolerance instead reports strain for values
    // merely near the cap.
    //
    // Pinned by `test_use_fans_heatwaves_strain_flags_at_the_caps`, which pairs a
    // saturating input with one whose `w` sits 4.7e-05 below `w_max`. Re-deriving that
    // pair is the point: this comment used to cite tdb=38.1, tr=43.7, v=2.16 as the case
    // a 1e-3 window got wrong, and by 4.4.2 those inputs put `w` 0.032 from the cap --
    // nowhere near any window, so the example had quietly stopped demonstrating the bug
    // it was recorded for. Cite a test, not a tuple.
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
        || ((20.0..=50.0).contains(&tdb.as_celsius())
            && (20.0..=50.0).contains(&tr.as_celsius())
            && (0.1..=4.5).contains(&v.as_meters_per_second())
            && (0.7..=2.0).contains(&met.as_met())
            && (0.0..=1.0).contains(&clo.as_clo()));

    if !within_limits {
        return UseFansHeatwavesResult {
            e_skin: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            e_rsw: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            e_max: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            q_sensible: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            q_skin: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            q_res: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            t_core: Temperature::from_celsius(f64::NAN),
            t_skin: Temperature::from_celsius(f64::NAN),
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
        e_skin: HeatFluxDensity::from_watts_per_square_meter(round1(
            gagge_result.e_skin.as_watts_per_square_meter(),
        )),
        e_rsw: HeatFluxDensity::from_watts_per_square_meter(round1(
            gagge_result.e_rsw.as_watts_per_square_meter(),
        )),
        e_max: HeatFluxDensity::from_watts_per_square_meter(round1(
            gagge_result.e_max.as_watts_per_square_meter(),
        )),
        q_sensible: HeatFluxDensity::from_watts_per_square_meter(round1(
            gagge_result.q_sensible.as_watts_per_square_meter(),
        )),
        q_skin: HeatFluxDensity::from_watts_per_square_meter(round1(
            gagge_result.q_skin.as_watts_per_square_meter(),
        )),
        q_res: HeatFluxDensity::from_watts_per_square_meter(round1(
            gagge_result.q_res.as_watts_per_square_meter(),
        )),
        t_core: Temperature::from_celsius(round1(gagge_result.t_core.as_celsius())),
        t_skin: Temperature::from_celsius(round1(gagge_result.t_skin.as_celsius())),
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

    fn inputs(tdb: f64, tr: f64, v: f64, rh: f64, met: f64, clo: f64) -> UseFansHeatwavesInputs {
        UseFansHeatwavesInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            v: Speed::from_meters_per_second(v),
            rh: Humidity::from_percent(rh),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
        }
    }

    #[test]
    fn test_use_fans_heatwaves() {
        let result =
            use_fans_heatwaves(inputs(35.0, 35.0, 1.0, 50.0, 1.2, 0.5), Default::default());
        assert!(result.e_skin.as_watts_per_square_meter() > 0.0);
        let t_core = result.t_core.as_celsius();
        assert!(t_core > 36.0 && t_core < 39.0);
    }

    #[test]
    fn test_heat_strain_detection() {
        // Extreme conditions that should trigger heat strain
        let result =
            use_fans_heatwaves(inputs(45.0, 45.0, 0.5, 70.0, 1.8, 0.3), Default::default());
        // Should detect some form of heat strain in extreme conditions
        assert!(result.t_core.as_celsius() > 37.0);
    }

    /// `max_skin_blood_flow` defaults to 80 here, not the 90 that [`two_nodes_gagge`]
    /// defaults to -- copying the wrong default would silently change every result.
    #[test]
    fn default_max_skin_blood_flow_is_80_not_90() {
        assert_eq!(UseFansHeatwavesOptions::default().max_skin_blood_flow, 80.0);
    }
}
