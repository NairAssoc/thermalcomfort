//! Simple thermal indices for heat and cold stress assessment
//!
//! This module contains implementations of various simple thermal comfort indices
//! that are widely used for quick assessment of thermal stress conditions.

use crate::psychrometrics::{PsyTaRhInputs, PsyTaRhOptions, dew_point_temperature, psy_ta_rh};
use measurements::{Humidity, Speed, Temperature};

/// The comfort inputs to [`wci`]: pythermalcomfort requires both (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WciInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Wind speed 10m above ground level
    pub v: Speed,
}

/// Optional parameters for [`wci`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WciOptions {
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for WciOptions {
    fn default() -> Self {
        Self { round_output: true }
    }
}

/// Calculate Wind Chill Index (WCI) - ASHRAE 2017
///
/// The wind chill index is an empirical index based on cooling measurements
/// taken on a cylindrical flask in Antarctica. It describes the rate of heat loss
/// as a function of ambient temperature and wind velocity.
///
/// # Returns
///
/// Wind Chill Index [W/m²]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{wci, WciInputs, WciOptions};
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = wci(
///     WciInputs {
///         tdb: Temperature::from_celsius(-5.0),
///         v: Speed::from_meters_per_second(5.5),
///     },
///     Default::default(),
/// );
/// assert!((result - 1255.2).abs() < 0.1);
/// ```
///
/// # References
///
/// - ASHRAE 2017 Handbook Fundamentals - Chapter 9
pub fn wci(inputs: WciInputs, options: WciOptions) -> f64 {
    let WciInputs { tdb, v } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let wind_speed_mps = v.as_meters_per_second();

    let mut wci_value =
        (10.45 + 10.0 * libm::sqrt(wind_speed_mps) - wind_speed_mps) * (33.0 - dry_bulb_celsius);

    // Convert to W/m²
    wci_value *= 1.163;

    if options.round_output {
        wci_value = crate::utilities::round_half_even(wci_value * 10.0) / 10.0;
    }

    wci_value
}

/// The comfort inputs to [`wind_chill_temperature`]: pythermalcomfort requires both (no
/// default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindChillTemperatureInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Wind speed 10m above ground level
    pub v: Speed,
}

/// Optional parameters for [`wind_chill_temperature`], with pythermalcomfort's
/// defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindChillTemperatureOptions {
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for WindChillTemperatureOptions {
    fn default() -> Self {
        Self { round_output: true }
    }
}

/// Calculate Wind Chill Temperature (WCT)
///
/// North American and United Kingdom wind chill index.
///
/// # Returns
///
/// Wind Chill Temperature [°C]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{
///     wind_chill_temperature, WindChillTemperatureInputs, WindChillTemperatureOptions,
/// };
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = wind_chill_temperature(
///     WindChillTemperatureInputs {
///         tdb: Temperature::from_celsius(-5.0),
///         v: Speed::from_kilometers_per_hour(5.5),
///     },
///     Default::default(),
/// );
/// assert!((result - (-7.5)).abs() < 0.1);
/// ```
pub fn wind_chill_temperature(
    inputs: WindChillTemperatureInputs,
    options: WindChillTemperatureOptions,
) -> f64 {
    let WindChillTemperatureInputs { tdb, v } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let wind_speed_kmh = v.as_kilometers_per_hour();

    let mut wct = 13.12 + 0.6215 * dry_bulb_celsius - 11.37 * libm::pow(wind_speed_kmh, 0.16)
        + 0.3965 * dry_bulb_celsius * libm::pow(wind_speed_kmh, 0.16);

    if options.round_output {
        wct = crate::utilities::round_half_even(wct * 10.0) / 10.0;
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

/// The comfort inputs to [`humidex`]: pythermalcomfort requires both (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumidexInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
}

/// Which vapor-pressure model [`humidex`] uses.
///
/// Matches pythermalcomfort's `model: str = "rana"` (`"rana"` / `"masterson"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HumidexModel {
    /// Rana et al. (2013), using the Magnus formula directly on `tdb`/`rh`. Upstream's
    /// default.
    #[default]
    Rana,
    /// Masterson and Richardson (1979), routed through dew point temperature.
    Masterson,
}

/// Optional parameters for [`humidex`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumidexOptions {
    /// Which vapor-pressure model to use
    pub model: HumidexModel,
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for HumidexOptions {
    fn default() -> Self {
        Self {
            model: HumidexModel::default(),
            round_output: true,
        }
    }
}

/// Calculate the Canadian Humidex
///
/// The humidex describes how hot, humid weather is felt by the average person.
/// It differs from the heat index in being related to the dew point rather than
/// relative humidity.
///
/// # Returns
///
/// [`HumidexResult`] with the humidex value [°C] and discomfort category.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{humidex, HumidexDiscomfort, HumidexInputs, HumidexOptions};
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = humidex(
///     HumidexInputs {
///         tdb: Temperature::from_celsius(25.0),
///         rh: Humidity::from_percent(50.0),
///     },
///     Default::default(),
/// );
/// assert!((result.humidex - 28.2).abs() < 0.2);
/// assert_eq!(result.discomfort, HumidexDiscomfort::LittleOrNone);
/// ```
///
/// # References
///
/// - Masterson and Richardson (1979)
pub fn humidex(inputs: HumidexInputs, options: HumidexOptions) -> HumidexResult {
    let HumidexInputs { tdb, rh } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();

    let mut hi = match options.model {
        HumidexModel::Rana => {
            // Rana et al. (2013) model. Vapor pressure via the Magnus formula:
            // - 6.112 hPa: reference saturation vapor pressure
            // - 7.5 and 237.7: Magnus formula coefficients
            let vapor_pressure = 6.112
                * libm::pow(10.0, 7.5 * dry_bulb_celsius / (237.7 + dry_bulb_celsius))
                * rh_percent
                / 100.0;
            // - 5/9: Fahrenheit to Celsius conversion factor
            // - 10.0 hPa: reference vapor pressure (comfort threshold)
            dry_bulb_celsius + 5.0 / 9.0 * (vapor_pressure - 10.0)
        }
        HumidexModel::Masterson => {
            let t_dp_celsius = dew_point_temperature(tdb, rh).as_celsius();
            let vapor_pressure =
                6.11 * libm::exp(5417.753 * (1.0 / 273.15 - 1.0 / (t_dp_celsius + 273.15)));
            dry_bulb_celsius + 5.0 / 9.0 * (vapor_pressure - 10.0)
        }
    };

    if options.round_output {
        hi = crate::utilities::round_half_even(hi * 10.0) / 10.0;
    }

    HumidexResult {
        humidex: hi,
        discomfort: HumidexDiscomfort::from_humidex(hi),
    }
}

/// The comfort inputs to [`thi`]: pythermalcomfort requires both (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThiInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
}

/// Optional parameters for [`thi`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThiOptions {
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for ThiOptions {
    fn default() -> Self {
        Self { round_output: true }
    }
}

/// Calculate Temperature-Humidity Index (THI)
///
/// # Returns
///
/// Temperature-Humidity Index
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{thi, ThiInputs, ThiOptions};
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = thi(
///     ThiInputs {
///         tdb: Temperature::from_celsius(25.0),
///         rh: Humidity::from_percent(50.0),
///     },
///     Default::default(),
/// );
/// assert!((result - 71.8).abs() < 0.2);
/// ```
pub fn thi(inputs: ThiInputs, options: ThiOptions) -> f64 {
    let ThiInputs { tdb, rh } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();

    let mut thi_value = 1.8 * dry_bulb_celsius + 32.0
        - 0.55 * (1.0 - 0.01 * rh_percent) * (1.8 * dry_bulb_celsius - 26.0);

    if options.round_output {
        thi_value = crate::utilities::round_half_even(thi_value * 10.0) / 10.0;
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

/// The comfort inputs to [`discomfort_index`]: pythermalcomfort requires both, and has
/// no optional parameters at all.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscomfortIndexInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
}

/// Calculate Discomfort Index (DI)
///
/// The index is essentially an effective temperature based on air temperature
/// and humidity. It only applies to warm environments.
///
/// # Returns
///
/// [`DiscomfortIndexResult`] with the DI value [°C] and discomfort condition.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{discomfort_index, DiscomfortCondition, DiscomfortIndexInputs};
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = discomfort_index(DiscomfortIndexInputs {
///     tdb: Temperature::from_celsius(25.0),
///     rh: Humidity::from_percent(50.0),
/// });
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
pub fn discomfort_index(inputs: DiscomfortIndexInputs) -> DiscomfortIndexResult {
    let DiscomfortIndexInputs { tdb, rh } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();

    let di = dry_bulb_celsius - 0.55 * (1.0 - 0.01 * rh_percent) * (dry_bulb_celsius - 14.5);
    // Categorise the raw value; pythermalcomfort maps the band before rounding, so a
    // di of 26.987 is "More than 50% feels discomfort" even though it prints as 27.0.
    let condition = DiscomfortCondition::from_di(di);
    let di = crate::utilities::round_half_even(di * 10.0) / 10.0;

    DiscomfortIndexResult {
        di,
        discomfort_condition: condition,
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
    pub fn from_hi_opt(hi: f64) -> Option<Self> {
        if hi.is_nan() {
            return None;
        }
        Some(Self::from_hi(hi))
    }

    /// Categorize a heat-index value that is known to be a number.
    ///
    /// Prefer [`from_hi_opt`](Self::from_hi_opt): a NaN heat index has no band, and
    /// falling through to `ExtremeDanger` reported the most severe category for a
    /// calculation that was never made.
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

/// The comfort inputs to [`heat_index_rothfusz`]: pythermalcomfort requires both (no
/// default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatIndexRothfuszInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
}

/// Optional parameters for [`heat_index_rothfusz`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatIndexRothfuszOptions {
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
    /// If true, returns NaN/None for tdb < 27°C
    pub limit_inputs: bool,
}

impl Default for HeatIndexRothfuszOptions {
    fn default() -> Self {
        Self {
            round_output: true,
            limit_inputs: true,
        }
    }
}

/// Calculate Heat Index using Rothfusz (1990) model
///
/// # Returns
///
/// [`HeatIndexResult`] with the heat index [°C] and stress category (None when
/// `limit_inputs` is true and the input falls below the applicability range).
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{
///     heat_index_rothfusz, HeatIndexRothfuszInputs, HeatIndexRothfuszOptions, HeatIndexStress,
/// };
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = heat_index_rothfusz(
///     HeatIndexRothfuszInputs {
///         tdb: Temperature::from_celsius(29.0),
///         rh: Humidity::from_percent(50.0),
///     },
///     Default::default(),
/// );
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
    inputs: HeatIndexRothfuszInputs,
    options: HeatIndexRothfuszOptions,
) -> HeatIndexResult {
    let HeatIndexRothfuszInputs { tdb, rh } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();

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
    if options.limit_inputs && dry_bulb_celsius < 27.0 {
        return HeatIndexResult {
            hi: f64::NAN,
            stress_category: None,
        };
    }

    if options.round_output {
        hi = crate::utilities::round_half_even(hi * 10.0) / 10.0;
    }

    HeatIndexResult {
        hi,
        stress_category: HeatIndexStress::from_hi_opt(hi),
    }
}

/// The comfort inputs to [`heat_index_schoen`]: pythermalcomfort requires both (no
/// default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatIndexSchoenInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
}

/// Optional parameters for [`heat_index_schoen`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatIndexSchoenOptions {
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for HeatIndexSchoenOptions {
    fn default() -> Self {
        Self { round_output: true }
    }
}

/// Calculate the Temperature Humidity Index (THI) using the Schoen (2005) model
///
/// Also known as the Heat Index Schoen. The THI is a simplified scale of apparent
/// temperature considering only dry-bulb temperature and humidity; it is another
/// formulation of the heat index.
///
/// # Returns
///
/// [`HeatIndexResult`] with the heat index [°C] and stress category. Unlike
/// [`heat_index_rothfusz`], this model has no dry-bulb applicability gate, so the
/// stress category is always populated.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{
///     heat_index_schoen, HeatIndexSchoenInputs, HeatIndexSchoenOptions, HeatIndexStress,
/// };
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = heat_index_schoen(
///     HeatIndexSchoenInputs {
///         tdb: Temperature::from_celsius(29.0),
///         rh: Humidity::from_percent(50.0),
///     },
///     Default::default(),
/// );
/// assert!((result.hi - 30.0).abs() < 0.2);
/// assert_eq!(result.stress_category, Some(HeatIndexStress::Caution));
/// ```
///
/// # References
///
/// - Schoen, C. (2005). A new empirical model of the temperature-humidity index.
///   Journal of Applied Meteorology, 44(9), 1413-1420.
pub fn heat_index_schoen(
    inputs: HeatIndexSchoenInputs,
    options: HeatIndexSchoenOptions,
) -> HeatIndexResult {
    let HeatIndexSchoenInputs { tdb, rh } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let t_dew_celsius = dew_point_temperature(tdb, rh).as_celsius();

    // Schoen (2005) empirical THI formulation:
    // hi = tdb - 1.0799 * exp(0.03755 * tdb) * (1 - exp(0.0801 * (t_dew - 14)))
    let mut hi = dry_bulb_celsius
        - 1.0799
            * libm::exp(0.03755 * dry_bulb_celsius)
            * (1.0 - libm::exp(0.0801 * (t_dew_celsius - 14.0)));

    if options.round_output {
        hi = crate::utilities::round_half_even(hi * 10.0) / 10.0;
    }

    HeatIndexResult {
        hi,
        stress_category: HeatIndexStress::from_hi_opt(hi),
    }
}

/// The comfort inputs to [`at`]: pythermalcomfort requires all three (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
    /// Wind speed 10m above ground level
    pub v: Speed,
}

/// Optional parameters for [`at`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtOptions {
    /// Net radiation absorbed per unit area of body surface [W/m²]
    pub q: Option<f64>,
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for AtOptions {
    fn default() -> Self {
        Self {
            q: None,
            round_output: true,
        }
    }
}

/// Calculate Apparent Temperature (AT)
///
/// The AT is defined as the temperature at the reference humidity level producing
/// the same amount of discomfort as that experienced under the current ambient
/// temperature, humidity, and solar radiation. It includes the chilling effect of
/// the wind at lower temperatures.
///
/// # Returns
///
/// Apparent Temperature [°C]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{at, AtInputs, AtOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity};
///
/// let result = at(
///     AtInputs {
///         tdb: Temperature::from_celsius(25.0),
///         rh: Humidity::from_percent(30.0),
///         v: Speed::from_meters_per_second(0.1),
///     },
///     Default::default(),
/// );
/// assert!((result - 24.1).abs() < 0.5);
/// ```
///
/// # References
///
/// - Steadman (1984)
/// - Australian Bureau of Meteorology
pub fn at(inputs: AtInputs, options: AtOptions) -> f64 {
    let AtInputs { tdb, rh, v } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let wind_speed_mps = v.as_meters_per_second();

    // Calculate vapor pressure using psychrometric function
    use measurements::Pressure;
    let psy_result = psy_ta_rh(
        PsyTaRhInputs { tdb, rh },
        PsyTaRhOptions {
            p_atm: Pressure::from_pascals(101325.0),
        },
    );
    let p_vap = psy_result.p_vap.as_pascals() / 100.0; // Convert to hPa

    // Calculate apparent temperature
    let mut t_at = if let Some(q_val) = options.q {
        // With solar radiation
        dry_bulb_celsius + 0.348 * p_vap - 0.7 * wind_speed_mps
            + 0.7 * q_val / (wind_speed_mps + 10.0)
            - 4.25
    } else {
        // Without solar radiation
        dry_bulb_celsius + 0.33 * p_vap - 0.7 * wind_speed_mps - 4.0
    };

    if options.round_output {
        t_at = crate::utilities::round_half_even(t_at * 10.0) / 10.0;
    }

    t_at
}

/// The comfort inputs to [`net`]: pythermalcomfort requires all three (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
    /// Wind speed at 1.2m above ground
    pub v: Speed,
}

/// Optional parameters for [`net`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetOptions {
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for NetOptions {
    fn default() -> Self {
        Self { round_output: true }
    }
}

/// Calculate Normal Effective Temperature (NET)
///
/// The NET establishes a link between the same condition of the organism's
/// thermoregulatory capability (warm and cold perception) and the surrounding
/// environment's temperature and humidity. It is calculated as a function of
/// air temperature, relative humidity, and wind speed.
///
/// # Returns
///
/// Normal Effective Temperature [°C]
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{net, NetInputs, NetOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity};
///
/// let result = net(
///     NetInputs {
///         tdb: Temperature::from_celsius(37.0),
///         rh: Humidity::from_percent(100.0),
///         v: Speed::from_meters_per_second(0.1),
///     },
///     Default::default(),
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
pub fn net(inputs: NetInputs, options: NetOptions) -> f64 {
    let NetInputs { tdb, rh, v } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let wind_speed_mps = v.as_meters_per_second();
    let rh_percent = rh.as_percent();

    let frac = 1.0 / (1.76 + 1.4 * libm::pow(wind_speed_mps, 0.75));
    let mut et = 37.0
        - (37.0 - dry_bulb_celsius) / (0.68 - 0.0014 * rh_percent + frac)
        - 0.29 * dry_bulb_celsius * (1.0 - 0.01 * rh_percent);

    if options.round_output {
        et = crate::utilities::round_half_even(et * 10.0) / 10.0;
    }

    et
}

/// The comfort inputs to [`esi`]: pythermalcomfort requires all three (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsiInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity (use `Humidity::from_percent()` for RH%)
    pub rh: Humidity,
    /// Global solar radiation [W/m²]
    pub sol_radiation_global: f64,
}

/// Optional parameters for [`esi`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsiOptions {
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for EsiOptions {
    fn default() -> Self {
        Self { round_output: true }
    }
}

/// Calculate Environmental Stress Index (ESI)
///
/// The ESI is an empirical index that combines temperature, humidity, and
/// solar radiation to assess heat stress.
///
/// # Returns
///
/// Environmental Stress Index
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::thermal_indices::{esi, EsiInputs, EsiOptions};
/// use thermalcomfort::{Temperature, Humidity};
///
/// let result = esi(
///     EsiInputs {
///         tdb: Temperature::from_celsius(30.2),
///         rh: Humidity::from_percent(42.2),
///         sol_radiation_global: 766.0,
///     },
///     Default::default(),
/// );
/// assert!((result - 26.2).abs() < 0.5);
/// ```
///
/// # References
///
/// - Moran et al. (2001)
pub fn esi(inputs: EsiInputs, options: EsiOptions) -> f64 {
    let EsiInputs {
        tdb,
        rh,
        sol_radiation_global,
    } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();

    let mut esi_value = 0.63 * dry_bulb_celsius - 0.03 * rh_percent
        + 0.002 * sol_radiation_global
        + 0.0054 * (dry_bulb_celsius * rh_percent)
        - 0.073 * libm::pow(0.1 + sol_radiation_global, -1.0);

    if options.round_output {
        esi_value = crate::utilities::round_half_even(esi_value * 10.0) / 10.0;
    }

    esi_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wci() {
        let result = wci(
            WciInputs {
                tdb: Temperature::from_celsius(-5.0),
                v: Speed::from_meters_per_second(5.5),
            },
            Default::default(),
        );
        assert!((result - 1255.2).abs() < 1.0);
    }

    #[test]
    fn test_wind_chill_temperature() {
        let result = wind_chill_temperature(
            WindChillTemperatureInputs {
                tdb: Temperature::from_celsius(-5.0),
                v: Speed::from_kilometers_per_hour(5.5),
            },
            Default::default(),
        );
        assert!((result - (-7.5)).abs() < 0.2);
    }

    #[test]
    fn test_humidex() {
        let result = humidex(
            HumidexInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(50.0),
            },
            Default::default(),
        );
        assert!((result.humidex - 28.2).abs() < 0.3);
        assert_eq!(result.discomfort, HumidexDiscomfort::LittleOrNone);
    }

    #[test]
    fn test_humidex_masterson_model() {
        // Masterson model is selected via HumidexOptions::model, not a separate function.
        let result = humidex(
            HumidexInputs {
                tdb: Temperature::from_celsius(30.0),
                rh: Humidity::from_percent(60.0),
            },
            HumidexOptions {
                model: HumidexModel::Masterson,
                round_output: true,
            },
        );
        let rana = humidex(
            HumidexInputs {
                tdb: Temperature::from_celsius(30.0),
                rh: Humidity::from_percent(60.0),
            },
            HumidexOptions {
                model: HumidexModel::Rana,
                round_output: true,
            },
        );
        // The two models use different vapor pressure formulas, so they should diverge.
        assert!((result.humidex - rana.humidex).abs() > 0.01);
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
            ThiInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(50.0),
            },
            Default::default(),
        );
        assert!((result - 71.8).abs() < 0.2);
    }

    #[test]
    fn test_discomfort_index() {
        let result = discomfort_index(DiscomfortIndexInputs {
            tdb: Temperature::from_celsius(25.0),
            rh: Humidity::from_percent(50.0),
        });
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
            HeatIndexRothfuszInputs {
                tdb: Temperature::from_celsius(29.0),
                rh: Humidity::from_percent(50.0),
            },
            Default::default(),
        );
        assert!((result.hi - 29.7).abs() < 0.5);
        assert_eq!(result.stress_category, Some(HeatIndexStress::Caution));

        // Below the applicability range with limit_inputs=true → NaN and no category
        let result = heat_index_rothfusz(
            HeatIndexRothfuszInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(50.0),
            },
            Default::default(),
        );
        assert!(result.hi.is_nan());
        assert_eq!(result.stress_category, None);

        // Same inputs with limits disabled → numeric value and a category
        let result = heat_index_rothfusz(
            HeatIndexRothfuszInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(50.0),
            },
            HeatIndexRothfuszOptions {
                round_output: true,
                limit_inputs: false,
            },
        );
        assert!(!result.hi.is_nan());
        assert!(result.stress_category.is_some());
    }

    #[test]
    fn test_heat_index_schoen() {
        // Reference values from pythermalcomfort 4.4.0 heat_index_schoen
        for (tdb, rh, expected_hi, expected_cat) in [
            (29.0, 50.0, 30.0, HeatIndexStress::Caution),
            (35.0, 60.0, 41.6, HeatIndexStress::Danger),
            (20.0, 40.0, 18.9, HeatIndexStress::NoRisk),
        ] {
            let result = heat_index_schoen(
                HeatIndexSchoenInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                Default::default(),
            );
            assert!(
                (result.hi - expected_hi).abs() < 0.05,
                "schoen({tdb}, {rh}) = {} expected {expected_hi}",
                result.hi
            );
            assert_eq!(result.stress_category, Some(expected_cat));
        }

        // Unlike Rothfusz, Schoen has no applicability gate: low tdb still yields a value
        let result = heat_index_schoen(
            HeatIndexSchoenInputs {
                tdb: Temperature::from_celsius(10.0),
                rh: Humidity::from_percent(50.0),
            },
            Default::default(),
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
            AtInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(30.0),
                v: Speed::from_meters_per_second(0.1),
            },
            Default::default(),
        );
        assert!((result - 24.1).abs() < 0.5);

        // Test with solar radiation
        let result = at(
            AtInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(30.0),
                v: Speed::from_meters_per_second(0.1),
            },
            AtOptions {
                q: Some(200.0),
                round_output: true,
            },
        );
        assert!((result - 37.9).abs() < 0.5);
    }

    #[test]
    fn test_net() {
        let result = net(
            NetInputs {
                tdb: Temperature::from_celsius(37.0),
                rh: Humidity::from_percent(100.0),
                v: Speed::from_meters_per_second(0.1),
            },
            Default::default(),
        );
        assert!((result - 37.0).abs() < 0.2);

        let result = net(
            NetInputs {
                tdb: Temperature::from_celsius(30.0),
                rh: Humidity::from_percent(60.0),
                v: Speed::from_meters_per_second(0.5),
            },
            NetOptions {
                round_output: false,
            },
        );
        assert!(result > 20.0 && result < 35.0);
    }

    #[test]
    fn test_esi() {
        let result = esi(
            EsiInputs {
                tdb: Temperature::from_celsius(30.2),
                rh: Humidity::from_percent(42.2),
                sol_radiation_global: 766.0,
            },
            Default::default(),
        );
        assert!((result - 26.2).abs() < 0.5);
    }
}
