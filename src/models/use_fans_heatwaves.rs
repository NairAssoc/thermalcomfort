//! Fan use during heatwaves assessment
//!
//! Estimate if environmental conditions would cause heat strain during heatwaves
//! when using fans.

use crate::models::two_nodes_gagge::{GaggeTwoNodesOptions, two_nodes_gagge};
use crate::utilities::Posture;
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// Result of fan use during heatwaves assessment
#[derive(Debug, Clone, Copy)]
pub struct UseFansHeatwavesResult {
    /// Heat loss from skin [W]
    pub e_skin: f64,
    /// Heat loss from regulatory sweating [W]
    pub e_rsw: f64,
    /// Maximum evaporative capacity [W]
    pub e_max: f64,
    /// Sensible heat loss [W]
    pub q_sensible: f64,
    /// Total heat loss from skin [W]
    pub q_skin: f64,
    /// Heat loss by respiration [W]
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
    pub heat_strain_blood_flow: bool,
    /// Heat strain from wettedness (w at maximum)
    pub heat_strain_w: bool,
    /// Heat strain from sweating (m_rsw at maximum)
    pub heat_strain_sweating: bool,
    /// Overall heat strain indicator
    pub heat_strain: bool,
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
/// * `metabolic_rate` - Metabolic rate [met, 0.7-2.0]
/// * `clothing_insulation` - Clothing insulation [clo, 0-1]
/// * `wme` - External work [met, default 0]
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
/// use thermalcomfort::{Temperature, Speed, Area, Pressure, Humidity};
///
/// let result = use_fans_heatwaves(
///     Temperature::from_celsius(35.0),
///     Temperature::from_celsius(35.0),
///     Speed::from_meters_per_second(1.0),
///     Humidity::from_percent(50.0),
///     1.2,
///     0.5,
///     0.0,
///     Area::from_square_meters(1.8258),
///     Pressure::from_pascals(101325.0),
///     Posture::Standing,
///     80.0,
///     500.0
/// );
/// assert!(result.e_skin > 0.0);
/// ```
pub fn use_fans_heatwaves(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    air_speed: Speed,
    relative_humidity: Humidity,
    metabolic_rate: f64,
    clothing_insulation: f64,
    wme: f64,
    body_surface_area: Area,
    p_atm: Pressure,
    posture: Posture,
    max_skin_blood_flow: f64,
    max_sweating: f64,
) -> UseFansHeatwavesResult {
    // Run two-nodes Gagge model
    let options = GaggeTwoNodesOptions {
        wme,
        body_surface_area,
        p_atm,
        posture,
        max_skin_blood_flow,
        max_sweating,
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

    // Detect heat strain conditions
    let heat_strain_blood_flow = (gagge_result.m_bl - max_skin_blood_flow).abs() < 0.01;
    let heat_strain_w = (gagge_result.w - gagge_result.w_max).abs() < 0.001;
    let heat_strain_sweating = (gagge_result.m_rsw - max_sweating).abs() < 0.01;
    let heat_strain = heat_strain_blood_flow || heat_strain_w || heat_strain_sweating;

    UseFansHeatwavesResult {
        e_skin: libm::round(gagge_result.e_skin * 10.0) / 10.0,
        e_rsw: libm::round(gagge_result.e_rsw * 10.0) / 10.0,
        e_max: libm::round(gagge_result.e_max * 10.0) / 10.0,
        q_sensible: libm::round(gagge_result.q_sensible * 10.0) / 10.0,
        q_skin: libm::round(gagge_result.q_skin * 10.0) / 10.0,
        q_res: libm::round(gagge_result.q_res * 10.0) / 10.0,
        t_core: libm::round(gagge_result.t_core * 10.0) / 10.0,
        t_skin: libm::round(gagge_result.t_skin * 10.0) / 10.0,
        m_bl: libm::round(gagge_result.m_bl * 10.0) / 10.0,
        m_rsw: libm::round(gagge_result.m_rsw * 10.0) / 10.0,
        w: libm::round(gagge_result.w * 1000.0) / 1000.0,
        w_max: libm::round(gagge_result.w_max * 1000.0) / 1000.0,
        heat_strain_blood_flow,
        heat_strain_w,
        heat_strain_sweating,
        heat_strain,
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
            1.2,
            0.5,
            0.0,
            Area::from_square_meters(1.8258),
            Pressure::from_pascals(101325.0),
            Posture::Standing,
            80.0,
            500.0,
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
            1.8,
            0.3,
            0.0,
            Area::from_square_meters(1.8258),
            Pressure::from_pascals(101325.0),
            Posture::Standing,
            80.0,
            500.0,
        );
        // Should detect some form of heat strain in extreme conditions
        assert!(result.t_core > 37.0);
    }
}
