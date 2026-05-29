//! Simple thermal indices for heat and cold stress assessment
//!
//! This module contains implementations of various simple thermal comfort indices
//! that are widely used for quick assessment of thermal stress conditions.

use crate::psychrometrics::{dew_point_temperature, psy_ta_rh};
use measurements::{Humidity, Speed, Temperature};

/// Calculate Wind Chill Index (WCI) - ASHRAE 2017
///
/// The wind chill index is an empirical index based on cooling measurements
/// taken on a cylindrical flask in Antarctica. It describes the rate of heat loss
/// as a function of ambient temperature and wind velocity.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `wind_speed` - Wind speed 10m above ground level
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// Wind Chill Index [W/m²]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::wci;
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = wci(
///     Temperature::from_celsius(-5.0),
///     Speed::from_meters_per_second(5.5),
///     true
/// );
/// assert!((result - 1255.2).abs() < 0.1);
/// ```
///
/// # References
///
/// - ASHRAE 2017 Handbook Fundamentals - Chapter 9
pub fn wci(dry_bulb_temp: Temperature, wind_speed: Speed, round_output: bool) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let wind_speed_mps = wind_speed.as_meters_per_second();

    let mut wci_value =
        (10.45 + 10.0 * libm::sqrt(wind_speed_mps) - wind_speed_mps) * (33.0 - dry_bulb_celsius);

    // Convert to W/m²
    wci_value *= 1.163;

    if round_output {
        wci_value = libm::round(wci_value * 10.0) / 10.0;
    }

    wci_value
}

/// Calculate Wind Chill Temperature (WCT)
///
/// North American and United Kingdom wind chill index.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `wind_speed` - Wind speed 10m above ground level (in km/h)
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// Wind Chill Temperature [°C]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::wind_chill_temperature;
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = wind_chill_temperature(
///     Temperature::from_celsius(-5.0),
///     Speed::from_kilometers_per_hour(5.5),
///     true
/// );
/// assert!((result - (-7.5)).abs() < 0.1);
/// ```
pub fn wind_chill_temperature(
    dry_bulb_temp: Temperature,
    wind_speed: Speed,
    round_output: bool,
) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let wind_speed_kmh = wind_speed.as_kilometers_per_hour();

    let mut wct = 13.12 + 0.6215 * dry_bulb_celsius - 11.37 * libm::pow(wind_speed_kmh, 0.16)
        + 0.3965 * dry_bulb_celsius * libm::pow(wind_speed_kmh, 0.16);

    if round_output {
        wct = libm::round(wct * 10.0) / 10.0;
    }

    wct
}

/// Humidex result with discomfort category
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumidexResult {
    /// Humidex value [°C]
    pub humidex: f64,
    /// Discomfort category derived from the humidex value
    pub discomfort: HumidexDiscomfort,
}

/// Discomfort categories for the humidex index.
///
/// Mapping follows Masterson and Richardson (1979) as used by pythermalcomfort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumidexDiscomfort {
    /// `hi <= 30`
    LittleOrNone,
    /// `30 < hi <= 35`
    Noticeable,
    /// `35 < hi <= 40`
    Evident,
    /// `40 < hi <= 45`
    Intense,
    /// `45 < hi <= 54`
    Dangerous,
    /// `hi > 54`
    HeatStrokeProbable,
}

impl HumidexDiscomfort {
    /// Categorize a humidex value.
    pub fn from_humidex(hi: f64) -> Self {
        if hi <= 30.0 {
            HumidexDiscomfort::LittleOrNone
        } else if hi <= 35.0 {
            HumidexDiscomfort::Noticeable
        } else if hi <= 40.0 {
            HumidexDiscomfort::Evident
        } else if hi <= 45.0 {
            HumidexDiscomfort::Intense
        } else if hi <= 54.0 {
            HumidexDiscomfort::Dangerous
        } else {
            HumidexDiscomfort::HeatStrokeProbable
        }
    }

    /// String form matching the pythermalcomfort `discomfort` field exactly.
    pub fn as_str(&self) -> &'static str {
        match self {
            HumidexDiscomfort::LittleOrNone => "Little or no discomfort",
            HumidexDiscomfort::Noticeable => "Noticeable discomfort",
            HumidexDiscomfort::Evident => "Evident discomfort",
            HumidexDiscomfort::Intense => "Intense discomfort; avoid exertion",
            HumidexDiscomfort::Dangerous => "Dangerous discomfort",
            HumidexDiscomfort::HeatStrokeProbable => "Heat stroke probable",
        }
    }
}

/// Calculate the Canadian Humidex
///
/// The humidex describes how hot, humid weather is felt by the average person.
/// It differs from the heat index in being related to the dew point rather than
/// relative humidity.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// [`HumidexResult`] with the humidex value [°C] and discomfort category.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{humidex, HumidexDiscomfort};
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = humidex(Temperature::from_celsius(25.0), Humidity::from_percent(50.0), true);
/// assert!((result.humidex - 28.2).abs() < 0.2);
/// assert_eq!(result.discomfort, HumidexDiscomfort::LittleOrNone);
/// ```
///
/// # References
///
/// - Masterson and Richardson (1979)
pub fn humidex(
    dry_bulb_temp: Temperature,
    relative_humidity: Humidity,
    round_output: bool,
) -> HumidexResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let rh_percent = relative_humidity.as_percent();

    // Rana et al. (2013) model
    // Vapor pressure calculation using Magnus formula:
    // - 6.112 hPa: reference saturation vapor pressure
    // - 7.5 and 237.7: Magnus formula coefficients
    let vapor_pressure =
        6.112 * libm::pow(10.0, 7.5 * dry_bulb_celsius / (237.7 + dry_bulb_celsius)) * rh_percent
            / 100.0;
    // Humidex calculation:
    // - 5/9: Fahrenheit to Celsius conversion factor
    // - 10.0 hPa: reference vapor pressure (comfort threshold)
    let mut hi = dry_bulb_celsius + 5.0 / 9.0 * (vapor_pressure - 10.0);

    if round_output {
        hi = libm::round(hi * 10.0) / 10.0;
    }

    HumidexResult {
        humidex: hi,
        discomfort: HumidexDiscomfort::from_humidex(hi),
    }
}

/// Calculate the Canadian Humidex (Masterson model)
///
/// Alternative humidex calculation using dew point temperature.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// Humidex value [°C]
pub fn humidex_masterson(
    dry_bulb_temp: Temperature,
    relative_humidity: Humidity,
    round_output: bool,
) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();

    let t_dp = dew_point_temperature(dry_bulb_temp, relative_humidity);
    let t_dp_celsius = t_dp.as_celsius();
    let vapor_pressure =
        6.11 * libm::exp(5417.753 * (1.0 / 273.15 - 1.0 / (t_dp_celsius + 273.15)));

    let mut hi = dry_bulb_celsius + 5.0 / 9.0 * (vapor_pressure - 10.0);

    if round_output {
        hi = libm::round(hi * 10.0) / 10.0;
    }

    hi
}

/// Calculate Temperature-Humidity Index (THI)
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// Temperature-Humidity Index
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::thi;
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = thi(Temperature::from_celsius(25.0), Humidity::from_percent(50.0), true);
/// assert!((result - 71.8).abs() < 0.2);
/// ```
pub fn thi(dry_bulb_temp: Temperature, relative_humidity: Humidity, round_output: bool) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let rh_percent = relative_humidity.as_percent();

    let mut thi_value = 1.8 * dry_bulb_celsius + 32.0
        - 0.55 * (1.0 - 0.01 * rh_percent) * (1.8 * dry_bulb_celsius - 26.0);

    if round_output {
        thi_value = libm::round(thi_value * 10.0) / 10.0;
    }

    thi_value
}

/// Discomfort Index result with categorical condition
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscomfortIndexResult {
    /// Discomfort Index [°C]
    pub di: f64,
    /// Discomfort condition derived from the index value
    pub discomfort_condition: DiscomfortCondition,
}

/// Discomfort categories for the Discomfort Index.
///
/// Bands per Polydoros (2015) as used by pythermalcomfort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscomfortCondition {
    /// `di < 21`
    NoDiscomfort,
    /// `21 <= di < 24`
    LessThan50PercentFeels,
    /// `24 <= di < 27`
    MoreThan50PercentFeels,
    /// `27 <= di < 29`
    MostFeelDiscomfort,
    /// `29 <= di < 32`
    EveryoneFeelsSevereStress,
    /// `di >= 32`
    MedicalEmergency,
}

impl DiscomfortCondition {
    /// Categorize a discomfort-index value.
    pub fn from_di(di: f64) -> Self {
        if di < 21.0 {
            DiscomfortCondition::NoDiscomfort
        } else if di < 24.0 {
            DiscomfortCondition::LessThan50PercentFeels
        } else if di < 27.0 {
            DiscomfortCondition::MoreThan50PercentFeels
        } else if di < 29.0 {
            DiscomfortCondition::MostFeelDiscomfort
        } else if di < 32.0 {
            DiscomfortCondition::EveryoneFeelsSevereStress
        } else {
            DiscomfortCondition::MedicalEmergency
        }
    }

    /// String form matching the pythermalcomfort `discomfort_condition` field exactly.
    pub fn as_str(&self) -> &'static str {
        match self {
            DiscomfortCondition::NoDiscomfort => "No discomfort",
            DiscomfortCondition::LessThan50PercentFeels => "Less than 50% feels discomfort",
            DiscomfortCondition::MoreThan50PercentFeels => "More than 50% feels discomfort",
            DiscomfortCondition::MostFeelDiscomfort => "Most of the population feels discomfort",
            DiscomfortCondition::EveryoneFeelsSevereStress => "Everyone feels severe stress",
            DiscomfortCondition::MedicalEmergency => "State of medical emergency",
        }
    }
}

/// Calculate Discomfort Index (DI)
///
/// The index is essentially an effective temperature based on air temperature
/// and humidity. It only applies to warm environments.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
///
/// # Returns
///
/// [`DiscomfortIndexResult`] with the DI value [°C] and discomfort condition.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{discomfort_index, DiscomfortCondition};
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = discomfort_index(Temperature::from_celsius(25.0), Humidity::from_percent(50.0));
/// assert!((result.di - 22.1).abs() < 0.1);
/// assert_eq!(result.discomfort_condition, DiscomfortCondition::LessThan50PercentFeels);
/// ```
///
/// # Discomfort Categories
///
/// - DI < 21°C: No discomfort
/// - 21 <= DI < 24°C: Less than 50% feels discomfort
/// - 24 <= DI < 27°C: More than 50% feels discomfort
/// - 27 <= DI < 29°C: Most of the population feels discomfort
/// - 29 <= DI < 32°C: Everyone feels severe stress
/// - DI >= 32°C: State of medical emergency
pub fn discomfort_index(
    dry_bulb_temp: Temperature,
    relative_humidity: Humidity,
) -> DiscomfortIndexResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let rh_percent = relative_humidity.as_percent();

    let di = dry_bulb_celsius - 0.55 * (1.0 - 0.01 * rh_percent) * (dry_bulb_celsius - 14.5);
    let di = libm::round(di * 10.0) / 10.0;

    DiscomfortIndexResult {
        di,
        discomfort_condition: DiscomfortCondition::from_di(di),
    }
}

/// Heat Index result with optional stress category.
///
/// `stress_category` is `Some(_)` when the model populates a category (e.g.,
/// Rothfusz) and `None` when it does not (e.g., Lu and Romps), matching the
/// `Optional[str]` field on Python's `HI` dataclass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatIndexResult {
    /// Heat Index [°C]
    pub hi: f64,
    /// Stress category derived from the heat index value
    pub stress_category: Option<HeatIndexStress>,
}

/// Heat Index stress categories per NWS / Rothfusz bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatIndexStress {
    /// `hi <= 27` — no risk
    NoRisk,
    /// `27 < hi <= 32` — caution
    Caution,
    /// `32 < hi <= 41` — extreme caution
    ExtremeCaution,
    /// `41 < hi <= 54` — danger
    Danger,
    /// `hi > 54` — extreme danger
    ExtremeDanger,
}

impl HeatIndexStress {
    /// Categorize a heat-index value. Bands are right-inclusive to match
    /// pythermalcomfort's `mapping(..., right=True)` semantics.
    pub fn from_hi(hi: f64) -> Self {
        if hi <= 27.0 {
            HeatIndexStress::NoRisk
        } else if hi <= 32.0 {
            HeatIndexStress::Caution
        } else if hi <= 41.0 {
            HeatIndexStress::ExtremeCaution
        } else if hi <= 54.0 {
            HeatIndexStress::Danger
        } else {
            HeatIndexStress::ExtremeDanger
        }
    }

    /// String form matching the pythermalcomfort `stress_category` field exactly.
    pub fn as_str(&self) -> &'static str {
        match self {
            HeatIndexStress::NoRisk => "no risk",
            HeatIndexStress::Caution => "caution",
            HeatIndexStress::ExtremeCaution => "extreme caution",
            HeatIndexStress::Danger => "danger",
            HeatIndexStress::ExtremeDanger => "extreme danger",
        }
    }
}

/// Calculate Heat Index using Rothfusz (1990) model
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `round_output` - Whether to round output to 1 decimal place
/// * `limit_inputs` - If true, returns NaN/None for tdb < 27°C
///
/// # Returns
///
/// [`HeatIndexResult`] with the heat index [°C] and stress category (None when
/// `limit_inputs` is true and the input falls below the applicability range).
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{heat_index_rothfusz, HeatIndexStress};
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = heat_index_rothfusz(Temperature::from_celsius(29.0), Humidity::from_percent(50.0), true, true);
/// assert!((result.hi - 29.7).abs() < 0.2);
/// assert_eq!(result.stress_category, Some(HeatIndexStress::Caution));
/// ```
///
/// # Heat Index Categories (right-inclusive)
///
/// - HI <= 27°C: no risk
/// - 27 < HI <= 32°C: caution
/// - 32 < HI <= 41°C: extreme caution
/// - 41 < HI <= 54°C: danger
/// - HI > 54°C: extreme danger
///
/// # References
///
/// - Rothfusz (1990) NWS Technical Attachment SR 90-23
pub fn heat_index_rothfusz(
    dry_bulb_temp: Temperature,
    relative_humidity: Humidity,
    round_output: bool,
    limit_inputs: bool,
) -> HeatIndexResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let rh_percent = relative_humidity.as_percent();

    // Rothfusz polynomial regression (Rothfusz 1990, NWS Technical Attachment SR 90-23)
    // All coefficients are empirically derived from regression analysis:
    // Constant: -8.784695
    // T coefficient: 1.61139411
    // RH coefficient: 2.338549
    // T×RH interaction: -0.14611605
    let mut hi = -8.784695 + 1.61139411 * dry_bulb_celsius + 2.338549 * rh_percent
        - 0.14611605 * dry_bulb_celsius * rh_percent;
    // Quadratic terms:
    // T² coefficient: -0.012308094
    // RH² coefficient: -0.016424828
    hi += -1.2308094e-2 * dry_bulb_celsius * dry_bulb_celsius
        - 1.6424828e-2 * rh_percent * rh_percent;
    // Higher-order interaction terms:
    // T²×RH coefficient: 0.002211732
    // T×RH² coefficient: 0.00072546
    hi += 2.211732e-3 * dry_bulb_celsius * dry_bulb_celsius * rh_percent
        + 7.2546e-4 * dry_bulb_celsius * rh_percent * rh_percent;
    // Highest-order term:
    // T²×RH² coefficient: -0.000003582
    hi += -3.582e-6 * dry_bulb_celsius * dry_bulb_celsius * rh_percent * rh_percent;

    // Heat index should only be calculated for temperatures above 27°C
    // This is the applicability limit from NWS (≈80°F)
    if limit_inputs && dry_bulb_celsius < 27.0 {
        return HeatIndexResult {
            hi: f64::NAN,
            stress_category: None,
        };
    }

    if round_output {
        hi = libm::round(hi * 10.0) / 10.0;
    }

    HeatIndexResult {
        hi,
        stress_category: Some(HeatIndexStress::from_hi(hi)),
    }
}

/// Calculate Apparent Temperature (AT)
///
/// The AT is defined as the temperature at the reference humidity level producing
/// the same amount of discomfort as that experienced under the current ambient
/// temperature, humidity, and solar radiation. It includes the chilling effect of
/// the wind at lower temperatures.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `wind_speed` - Wind speed 10m above ground level
/// * `q` - Net radiation absorbed per unit area of body surface [W/m²] (optional)
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// Apparent Temperature [°C]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::at;
/// use thermalcomfort::{Temperature, Speed, Humidity};
///
/// let result = at(
///     Temperature::from_celsius(25.0),
///     Humidity::from_percent(30.0),
///     Speed::from_meters_per_second(0.1),
///     None,
///     true
/// );
/// assert!((result - 24.1).abs() < 0.5);
/// ```
///
/// # References
///
/// - Steadman (1984)
/// - Australian Bureau of Meteorology
pub fn at(
    dry_bulb_temp: Temperature,
    relative_humidity: Humidity,
    wind_speed: Speed,
    q: Option<f64>,
    round_output: bool,
) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let wind_speed_mps = wind_speed.as_meters_per_second();

    // Calculate vapor pressure using psychrometric function
    use measurements::Pressure;
    let psy_result = psy_ta_rh(
        dry_bulb_temp,
        relative_humidity,
        Pressure::from_pascals(101325.0),
    );
    let p_vap = psy_result.p_vap.as_pascals() / 100.0; // Convert to hPa

    // Calculate apparent temperature
    let mut t_at = if let Some(q_val) = q {
        // With solar radiation
        dry_bulb_celsius + 0.348 * p_vap - 0.7 * wind_speed_mps
            + 0.7 * q_val / (wind_speed_mps + 10.0)
            - 4.25
    } else {
        // Without solar radiation
        dry_bulb_celsius + 0.33 * p_vap - 0.7 * wind_speed_mps - 4.0
    };

    if round_output {
        t_at = libm::round(t_at * 10.0) / 10.0;
    }

    t_at
}

/// Calculate Normal Effective Temperature (NET)
///
/// The NET establishes a link between the same condition of the organism's
/// thermoregulatory capability (warm and cold perception) and the surrounding
/// environment's temperature and humidity. It is calculated as a function of
/// air temperature, relative humidity, and wind speed.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `wind_speed` - Wind speed at 1.2m above ground
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// Normal Effective Temperature [°C]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::net;
/// use thermalcomfort::{Temperature, Speed, Humidity};
///
/// let result = net(
///     Temperature::from_celsius(37.0),
///     Humidity::from_percent(100.0),
///     Speed::from_meters_per_second(0.1),
///     true
/// );
/// assert!((result - 37.0).abs() < 0.1);
/// ```
///
/// # Thresholds (Central Europe)
///
/// - < 1°C: Very cold
/// - 1-9°C: Cold
/// - 9-17°C: Cool
/// - 17-21°C: Fresh
/// - 21-23°C: Comfortable
/// - 23-27°C: Warm
/// - > 27°C: Hot
///
/// # References
///
/// - Missenard (1933)
/// - Used in Germany and Hong Kong Observatory
pub fn net(
    dry_bulb_temp: Temperature,
    relative_humidity: Humidity,
    wind_speed: Speed,
    round_output: bool,
) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let wind_speed_mps = wind_speed.as_meters_per_second();
    let rh_percent = relative_humidity.as_percent();

    let frac = 1.0 / (1.76 + 1.4 * libm::pow(wind_speed_mps, 0.75));
    let mut et = 37.0
        - (37.0 - dry_bulb_celsius) / (0.68 - 0.0014 * rh_percent + frac)
        - 0.29 * dry_bulb_celsius * (1.0 - 0.01 * rh_percent);

    if round_output {
        et = libm::round(et * 10.0) / 10.0;
    }

    et
}

/// Calculate Environmental Stress Index (ESI)
///
/// The ESI is an empirical index that combines temperature, humidity, and
/// solar radiation to assess heat stress.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `sol_radiation_global` - Global solar radiation [W/m²]
/// * `round_output` - Whether to round output to 1 decimal place
///
/// # Returns
///
/// Environmental Stress Index
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::esi;
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = esi(Temperature::from_celsius(30.2), Humidity::from_percent(42.2), 766.0, true);
/// assert!((result - 26.2).abs() < 0.5);
/// ```
///
/// # References
///
/// - Moran et al. (2001)
pub fn esi(
    dry_bulb_temp: Temperature,
    relative_humidity: Humidity,
    sol_radiation_global: f64,
    round_output: bool,
) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let rh_percent = relative_humidity.as_percent();

    let mut esi_value = 0.63 * dry_bulb_celsius - 0.03 * rh_percent
        + 0.002 * sol_radiation_global
        + 0.0054 * (dry_bulb_celsius * rh_percent)
        - 0.073 * libm::pow(0.1 + sol_radiation_global, -1.0);

    if round_output {
        esi_value = libm::round(esi_value * 10.0) / 10.0;
    }

    esi_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wci() {
        let result = wci(
            Temperature::from_celsius(-5.0),
            Speed::from_meters_per_second(5.5),
            true,
        );
        assert!((result - 1255.2).abs() < 1.0);
    }

    #[test]
    fn test_wind_chill_temperature() {
        let result = wind_chill_temperature(
            Temperature::from_celsius(-5.0),
            Speed::from_kilometers_per_hour(5.5),
            true,
        );
        assert!((result - (-7.5)).abs() < 0.2);
    }

    #[test]
    fn test_humidex() {
        let result = humidex(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
            true,
        );
        assert!((result.humidex - 28.2).abs() < 0.3);
        assert_eq!(result.discomfort, HumidexDiscomfort::LittleOrNone);
    }

    #[test]
    fn test_humidex_categories() {
        // Boundaries: <=30 LittleOrNone, <=35 Noticeable, <=40 Evident,
        // <=45 Intense, <=54 Dangerous, >54 HeatStrokeProbable.
        assert_eq!(
            HumidexDiscomfort::from_humidex(30.0),
            HumidexDiscomfort::LittleOrNone
        );
        assert_eq!(
            HumidexDiscomfort::from_humidex(30.0001),
            HumidexDiscomfort::Noticeable
        );
        assert_eq!(
            HumidexDiscomfort::from_humidex(38.0),
            HumidexDiscomfort::Evident
        );
        assert_eq!(
            HumidexDiscomfort::from_humidex(45.0),
            HumidexDiscomfort::Intense
        );
        assert_eq!(
            HumidexDiscomfort::from_humidex(54.0),
            HumidexDiscomfort::Dangerous
        );
        assert_eq!(
            HumidexDiscomfort::from_humidex(60.0),
            HumidexDiscomfort::HeatStrokeProbable
        );
    }

    #[test]
    fn test_thi() {
        let result = thi(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
            true,
        );
        assert!((result - 71.8).abs() < 0.2);
    }

    #[test]
    fn test_discomfort_index() {
        let result = discomfort_index(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
        );
        assert!((result.di - 22.1).abs() < 0.2);
        assert_eq!(
            result.discomfort_condition,
            DiscomfortCondition::LessThan50PercentFeels
        );
    }

    #[test]
    fn test_discomfort_index_categories() {
        // Bands are left-inclusive: [21,24) [24,27) [27,29) [29,32) [32,inf)
        assert_eq!(
            DiscomfortCondition::from_di(20.9),
            DiscomfortCondition::NoDiscomfort
        );
        assert_eq!(
            DiscomfortCondition::from_di(21.0),
            DiscomfortCondition::LessThan50PercentFeels
        );
        assert_eq!(
            DiscomfortCondition::from_di(24.0),
            DiscomfortCondition::MoreThan50PercentFeels
        );
        assert_eq!(
            DiscomfortCondition::from_di(27.0),
            DiscomfortCondition::MostFeelDiscomfort
        );
        assert_eq!(
            DiscomfortCondition::from_di(29.0),
            DiscomfortCondition::EveryoneFeelsSevereStress
        );
        assert_eq!(
            DiscomfortCondition::from_di(32.0),
            DiscomfortCondition::MedicalEmergency
        );
    }

    #[test]
    fn test_heat_index_rothfusz() {
        let result = heat_index_rothfusz(
            Temperature::from_celsius(29.0),
            Humidity::from_percent(50.0),
            true,
            true,
        );
        assert!((result.hi - 29.7).abs() < 0.5);
        assert_eq!(result.stress_category, Some(HeatIndexStress::Caution));

        // Below the applicability range with limit_inputs=true → NaN and no category
        let result = heat_index_rothfusz(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
            true,
            true,
        );
        assert!(result.hi.is_nan());
        assert_eq!(result.stress_category, None);

        // Same inputs with limits disabled → numeric value and a category
        let result = heat_index_rothfusz(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
            true,
            false,
        );
        assert!(!result.hi.is_nan());
        assert!(result.stress_category.is_some());
    }

    #[test]
    fn test_heat_index_stress_categories() {
        // Right-inclusive bands per pythermalcomfort: <=27, <=32, <=41, <=54, >54.
        assert_eq!(HeatIndexStress::from_hi(27.0), HeatIndexStress::NoRisk);
        assert_eq!(HeatIndexStress::from_hi(27.1), HeatIndexStress::Caution);
        assert_eq!(HeatIndexStress::from_hi(32.0), HeatIndexStress::Caution);
        assert_eq!(
            HeatIndexStress::from_hi(32.1),
            HeatIndexStress::ExtremeCaution
        );
        assert_eq!(
            HeatIndexStress::from_hi(41.0),
            HeatIndexStress::ExtremeCaution
        );
        assert_eq!(HeatIndexStress::from_hi(41.1), HeatIndexStress::Danger);
        assert_eq!(HeatIndexStress::from_hi(54.0), HeatIndexStress::Danger);
        assert_eq!(
            HeatIndexStress::from_hi(54.1),
            HeatIndexStress::ExtremeDanger
        );
    }

    #[test]
    fn test_at() {
        // Test without solar radiation
        let result = at(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(30.0),
            Speed::from_meters_per_second(0.1),
            None,
            true,
        );
        assert!((result - 24.1).abs() < 0.5);

        // Test with solar radiation
        let result = at(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(30.0),
            Speed::from_meters_per_second(0.1),
            Some(200.0),
            true,
        );
        assert!((result - 37.9).abs() < 0.5);
    }

    #[test]
    fn test_net() {
        let result = net(
            Temperature::from_celsius(37.0),
            Humidity::from_percent(100.0),
            Speed::from_meters_per_second(0.1),
            true,
        );
        assert!((result - 37.0).abs() < 0.2);

        let result = net(
            Temperature::from_celsius(30.0),
            Humidity::from_percent(60.0),
            Speed::from_meters_per_second(0.5),
            false,
        );
        assert!(result > 20.0 && result < 35.0);
    }

    #[test]
    fn test_esi() {
        let result = esi(
            Temperature::from_celsius(30.2),
            Humidity::from_percent(42.2),
            766.0,
            true,
        );
        assert!((result - 26.2).abs() < 0.5);
    }
}
