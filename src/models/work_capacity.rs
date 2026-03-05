//! Work capacity models for heat stress assessment
//!
//! These models estimate the reduction in work capacity due to heat stress,
//! expressed as a percentage where 100% means work is unaffected by heat
//! and 0% means no work can be performed.

use measurements::{Power, Temperature};

/// Work intensity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkIntensity {
    /// Light work (< 200 kcal/h or ~200W)
    Light,
    /// Moderate work (200-350 kcal/h or ~300W)
    Moderate,
    /// Heavy work (350-500 kcal/h or ~400W)
    #[default]
    Heavy,
}

/// Estimate work capacity based on ISO standards
///
/// Estimates work capacity as described by Bröde et al. (2018).
/// Based on the relationship between WBGT and metabolic rate.
///
/// # Arguments
///
/// * `wbgt` - Wet Bulb Globe Temperature (use `Temperature::from_celsius()` or similar)
/// * `metabolic_power` - Metabolic heat production (use `Power::from_watts()` or similar)
///
/// # Returns
///
/// Work capacity as percentage (0-100%)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::work_capacity::work_capacity_iso;
/// use thermalcomfort::{Temperature, Power};
///
/// let capacity = work_capacity_iso(Temperature::from_celsius(30.0), Power::from_watts(300.0));
/// assert!(capacity >= 0.0 && capacity <= 100.0);
/// ```
///
/// # References
///
/// - Bröde P, Fiala D, Lemke B, Kjellstrom T (2018) Int J Biometeorol 62(3):331-45
pub fn work_capacity_iso(wbgt: Temperature, metabolic_power: Power) -> f64 {
    let wbgt_celsius = wbgt.as_celsius();
    let met = metabolic_power.as_watts();
    let met_rest = 117.0; // Assumed resting metabolic rate

    let wbgt_lim = 34.9 - met / 46.0;
    let wbgt_lim_rest = 34.9 - met_rest / 46.0;

    let capacity = ((wbgt_lim_rest - wbgt_celsius) / (wbgt_lim_rest - wbgt_lim)) * 100.0;

    // Clip to 0-100 range
    capacity.clamp(0.0, 100.0)
}

/// Estimate work capacity based on NIOSH standards
///
/// Estimates work capacity as described by Bröde et al. (2018).
/// Uses logarithmic relationship with metabolic rate.
///
/// # Arguments
///
/// * `wbgt` - Wet Bulb Globe Temperature (use `Temperature::from_celsius()` or similar)
/// * `metabolic_power` - Metabolic heat production (use `Power::from_watts()` or similar)
///
/// # Returns
///
/// Work capacity as percentage (0-100%)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::work_capacity::work_capacity_niosh;
/// use thermalcomfort::{Temperature, Power};
///
/// let capacity = work_capacity_niosh(Temperature::from_celsius(30.0), Power::from_watts(300.0));
/// assert!(capacity >= 0.0 && capacity <= 100.0);
/// ```
///
/// # References
///
/// - Bröde P, Fiala D, Lemke B, Kjellstrom T (2018) Int J Biometeorol 62(3):331-45
pub fn work_capacity_niosh(wbgt: Temperature, metabolic_power: Power) -> f64 {
    let wbgt_celsius = wbgt.as_celsius();
    let met = metabolic_power.as_watts();
    let met_rest = 117.0; // Assumed resting metabolic rate

    let wbgt_lim = 56.7 - 11.5 * libm::log10(met);
    let wbgt_lim_rest = 56.7 - 11.5 * libm::log10(met_rest);

    let capacity = ((wbgt_lim_rest - wbgt_celsius) / (wbgt_lim_rest - wbgt_lim)) * 100.0;

    // Clip to 0-100 range
    capacity.clamp(0.0, 100.0)
}

/// Estimate work capacity based on Dunne et al. (2013)
///
/// Based on NIOSH safety standards with intensity-specific multipliers.
///
/// # Arguments
///
/// * `wbgt` - Wet Bulb Globe Temperature (use `Temperature::from_celsius()` or similar)
/// * `work_intensity` - Intensity of work being performed
///
/// # Returns
///
/// Work capacity as percentage (0-100%)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::work_capacity::{work_capacity_dunne, WorkIntensity};
/// use thermalcomfort::Temperature;
///
/// let capacity = work_capacity_dunne(Temperature::from_celsius(30.0), WorkIntensity::Heavy);
/// assert!(capacity >= 0.0 && capacity <= 100.0);
/// ```
///
/// # References
///
/// - Dunne JP, Stouffer RJ, John JG (2013) Nature Climate Change 3(6):563-6
pub fn work_capacity_dunne(wbgt: Temperature, work_intensity: WorkIntensity) -> f64 {
    let wbgt_celsius = wbgt.as_celsius();
    // Base capacity calculation
    let wbgt_excess = (wbgt_celsius - 25.0).max(0.0);
    let mut capacity = 100.0 - 25.0 * libm::pow(wbgt_excess, 2.0 / 3.0);
    capacity = capacity.clamp(0.0, 100.0);

    // Apply intensity-specific multiplier
    let factor = match work_intensity {
        WorkIntensity::Heavy => 1.0,
        WorkIntensity::Moderate => 2.0,
        WorkIntensity::Light => 4.0,
    };

    (capacity * factor).min(100.0)
}

/// Estimate work capacity based on HOTHAPS (Kjellstrom et al. 2018)
///
/// Uses sigmoid function with intensity-specific parameters.
/// Note: Capacity never reaches 0% as it assumes at least 10% work is always possible
/// in short bursts.
///
/// # Arguments
///
/// * `wbgt` - Wet Bulb Globe Temperature (use `Temperature::from_celsius()` or similar)
/// * `work_intensity` - Intensity of work being performed
///
/// # Returns
///
/// Work capacity as percentage (10-100%)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::work_capacity::{work_capacity_hothaps, WorkIntensity};
/// use thermalcomfort::Temperature;
///
/// let capacity = work_capacity_hothaps(Temperature::from_celsius(30.0), WorkIntensity::Heavy);
/// assert!(capacity >= 10.0 && capacity <= 100.0);
/// ```
///
/// # References
///
/// - Kjellstrom T et al. (2018)
/// - Bröde P, Fiala D, Lemke B, Kjellstrom T (2018) Int J Biometeorol 62(3):331-45
pub fn work_capacity_hothaps(wbgt: Temperature, work_intensity: WorkIntensity) -> f64 {
    let wbgt_celsius = wbgt.as_celsius();
    let (divisor, exponent) = match work_intensity {
        WorkIntensity::Heavy => (30.94, 16.64),
        WorkIntensity::Moderate => (32.93, 17.81),
        WorkIntensity::Light => (34.64, 22.72),
    };

    let sigmoid = 0.9 / (1.0 + libm::pow(wbgt_celsius / divisor, exponent));
    let capacity = 100.0 * (0.1 + sigmoid);

    capacity.clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_capacity_iso() {
        // Moderate conditions
        let capacity = work_capacity_iso(Temperature::from_celsius(30.0), Power::from_watts(300.0));
        assert!(capacity > 0.0 && capacity <= 100.0);

        // Hot conditions - capacity should be lower
        let capacity_hot = work_capacity_iso(Temperature::from_celsius(35.0), Power::from_watts(300.0));
        assert!(capacity_hot < capacity);
    }

    #[test]
    fn test_work_capacity_niosh() {
        let capacity = work_capacity_niosh(Temperature::from_celsius(30.0), Power::from_watts(300.0));
        assert!(capacity > 0.0 && capacity <= 100.0);

        // Higher metabolic rate should reduce capacity limits
        let capacity_high_met = work_capacity_niosh(Temperature::from_celsius(30.0), Power::from_watts(400.0));
        assert!(capacity_high_met < capacity);
    }

    #[test]
    fn test_work_capacity_dunne() {
        // Test different intensities
        let heavy = work_capacity_dunne(Temperature::from_celsius(30.0), WorkIntensity::Heavy);
        let moderate =
            work_capacity_dunne(Temperature::from_celsius(30.0), WorkIntensity::Moderate);
        let light = work_capacity_dunne(Temperature::from_celsius(30.0), WorkIntensity::Light);

        assert!((0.0..=100.0).contains(&heavy));
        assert!(light >= moderate && moderate >= heavy);

        // Below 25°C should give 100% capacity
        let cool = work_capacity_dunne(Temperature::from_celsius(20.0), WorkIntensity::Heavy);
        assert!((cool - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_work_capacity_hothaps() {
        let capacity = work_capacity_hothaps(Temperature::from_celsius(30.0), WorkIntensity::Heavy);
        assert!((10.0..=100.0).contains(&capacity));

        // Light work should have higher capacity
        let light = work_capacity_hothaps(Temperature::from_celsius(30.0), WorkIntensity::Light);
        let heavy = work_capacity_hothaps(Temperature::from_celsius(30.0), WorkIntensity::Heavy);
        assert!(light > heavy);
    }
}
