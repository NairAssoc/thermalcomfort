//! Environmental quantities: mean radiant, operative and running-mean outdoor
//! temperatures, relative air speed, the logarithmic wind profile, and the solar-geometry
//! helpers. Mirrors `pythermalcomfort.environment`.

use crate::MetabolicRate;
use crate::constants::*;
use crate::utilities::{Units, np_maximum, round_to, round_to_exact_decimal};
use libm::{fabs as abs, log, pow, round, sqrt};
use measurements::{Angle, Length, Speed, Temperature};

/// Optional parameters for [`running_mean_outdoor_temperature`], with pythermalcomfort's
/// defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunningMeanOutdoorTemperatureOptions {
    /// Weighting constant between 0 and 1 (default: 0.8)
    /// - EN 16798-1 recommends 0.8
    /// - ASHRAE 55 recommends 0.6-0.9 (slow to fast response)
    /// - Use 0.9 for stable climates, 0.6 for variable climates
    pub alpha: f64,
    /// Unit system the result is expressed in.
    ///
    /// `utilities.py:944-950`'s IP branch is a genuine °C<->°F conversion: it converts
    /// each element of `temp_array` from °F before averaging, and converts the result
    /// back to °F afterwards, **rounding in that output unit**. Typed [`Temperature`]
    /// inputs make the input-side conversion moot (they already carry their true SI
    /// value regardless of `units`), but the output-side rounding is real: `round(t_rm_f,
    /// 1)` is not the same physical temperature as rounding an already-Celsius-rounded
    /// value and converting it, so it is replicated here rather than derived after the
    /// fact.
    pub units: Units,
}

impl Default for RunningMeanOutdoorTemperatureOptions {
    fn default() -> Self {
        Self {
            alpha: 0.8,
            units: Units::SI,
        }
    }
}

/// Calculate running mean outdoor temperature (prevailing mean)
///
/// Estimates the exponentially weighted running mean temperature from an array
/// of daily mean temperatures. Used by adaptive comfort models.
///
/// # Arguments
///
/// * `temp_array` - Array of daily mean temperatures in descending order
///   (newest/yesterday first: [t_day-1, t_day-2, ..., t_day-n])
/// * `options` - Weighting constant and output unit system
///
/// # Returns
///
/// Running mean outdoor temperature
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{running_mean_outdoor_temperature, RunningMeanOutdoorTemperatureOptions};
/// use thermalcomfort::Temperature;
///
/// // Last 7 days of daily mean temperatures (yesterday to 7 days ago)
/// let temps = vec![
///     Temperature::from_celsius(22.0),
///     Temperature::from_celsius(20.5),
///     Temperature::from_celsius(19.0),
///     Temperature::from_celsius(21.0),
///     Temperature::from_celsius(18.5),
///     Temperature::from_celsius(17.0),
///     Temperature::from_celsius(16.5),
/// ];
/// let t_rm = running_mean_outdoor_temperature(&temps, RunningMeanOutdoorTemperatureOptions::default());
/// // Rounded to one decimal, matching pythermalcomfort
/// assert!((t_rm.as_celsius() - 19.9).abs() < 0.01);
/// ```
pub fn running_mean_outdoor_temperature(
    temp_array: &[Temperature],
    options: RunningMeanOutdoorTemperatureOptions,
) -> Temperature {
    let RunningMeanOutdoorTemperatureOptions { alpha, units } = options;

    // utilities.py:944-950's IP branch converts an empty list identically to the SI
    // branch (both would hit ZeroDivisionError in Python); this NaN-avoiding shortcut is
    // a pre-existing Rust-only guard, not upstream behaviour, so it is unit-agnostic too.
    if temp_array.is_empty() {
        return match units {
            Units::SI => Temperature::from_celsius(0.0),
            Units::IP => Temperature::from_fahrenheit(32.0),
        };
    }

    let mut sum_weighted = 0.0;
    let mut sum_weights = 0.0;

    for (i, temp) in temp_array.iter().enumerate() {
        let weight = pow(alpha, i as f64);
        sum_weighted += weight * temp.as_celsius();
        sum_weights += weight;
    }

    let t_rm_celsius = sum_weighted / sum_weights;

    // utilities.py:944-950: converts to the output unit before rounding, matching
    // `units_converter`'s literal `(value * 9 / 5) + 32` formula exactly (rather than
    // routing through `Temperature`'s Kelvin storage, which would lose the last ULP —
    // see 28adfa8).
    let t_rm_out = match units {
        Units::SI => t_rm_celsius,
        Units::IP => t_rm_celsius * 9.0 / 5.0 + 32.0,
    };

    // pythermalcomfort rounds this to one decimal before returning; without it the two
    // implementations disagree in the second decimal for every input. `utilities.py:955`
    // rounds a plain Python float (not a numba-jit or numpy value), so this is CPython's
    // builtin decimal rounding, not numpy's.
    let t_rm_rounded = round_to_exact_decimal(t_rm_out, 1);

    match units {
        Units::SI => Temperature::from_celsius(t_rm_rounded),
        Units::IP => Temperature::from_fahrenheit(t_rm_rounded),
    }
}
/// Calculate relative air speed which combines average air speed plus body movement
///
/// # Arguments
///
/// * `v` - Air speed measured by sensor (use `Speed::from_meters_per_second()` or similar)
/// * `met` - Metabolic rate
///
/// # Returns
///
/// Relative air speed
///
/// # Example
///
/// ```
/// use thermalcomfort::environment::v_relative;
/// use thermalcomfort::{Speed, MetabolicRate};
///
/// let v = Speed::from_meters_per_second(0.1);
/// let met = MetabolicRate::from_met(1.4);
/// let vr = v_relative(v, met);
/// assert!((vr.as_meters_per_second() - 0.22).abs() < 0.01);
/// ```
#[inline]
pub fn v_relative(v: Speed, met: MetabolicRate) -> Speed {
    let v_ms = v.as_meters_per_second();
    let met_val = met.as_met();
    let vr_ms = if met_val > 1.0 {
        // Relative air speed accounts for increased air movement from body motion
        // 0.3 m/s per met above 1.0 (ISO 7730 and ASHRAE 55)
        // Round to 3 decimal places
        round((v_ms + 0.3 * (met_val - 1.0)) * 1000.0) / 1000.0
    } else {
        v_ms
    };
    Speed::from_meters_per_second(vr_ms)
}
/// The comfort inputs to [`mean_radiant_temperature`]: pythermalcomfort requires all
/// three (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanRadiantTemperatureInputs {
    /// Globe temperature
    pub tg: Temperature,
    /// Air temperature
    pub tdb: Temperature,
    /// Air speed
    pub v: Speed,
}

/// Optional parameters for [`mean_radiant_temperature`], with pythermalcomfort's
/// defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanRadiantTemperatureOptions {
    /// Globe diameter, default 0.15 m
    pub d: Length,
    /// Globe emissivity, default 0.95
    pub emissivity: f64,
    /// If true, use ISO 7726:1998 formula; if false, use Mixed Convection.
    ///
    /// Matches pythermalcomfort's `standard` string ("Mixed Convection" / "ISO");
    /// defaults to `false` (Mixed Convection).
    pub use_iso: bool,
}

impl Default for MeanRadiantTemperatureOptions {
    fn default() -> Self {
        Self {
            d: Length::from_meters(0.15),
            emissivity: 0.95,
            use_iso: false,
        }
    }
}

/// Calculate mean radiant temperature from globe temperature
///
/// Converts globe temperature reading to mean radiant temperature using either
/// Mixed Convection or ISO 7726:1998 standard
///
/// # Returns
///
/// Mean radiant temperature
///
/// # Note
///
/// The Mixed Convection formulation by Teitelbaum et al. (2022) is only
/// validated for globe diameters between 0.04 and 0.15 m
pub fn mean_radiant_temperature(
    inputs: MeanRadiantTemperatureInputs,
    options: MeanRadiantTemperatureOptions,
) -> Temperature {
    let MeanRadiantTemperatureInputs { tg, tdb, v } = inputs;
    let tg_celsius = tg.as_celsius();
    let tdb_celsius = tdb.as_celsius();
    let v_ms = v.as_meters_per_second();
    let d_m = options.d.as_meters();

    let tr_celsius = if options.use_iso {
        mean_radiant_temperature_iso(tg_celsius, tdb_celsius, v_ms, d_m, options.emissivity)
    } else {
        mean_radiant_temperature_mixed(tg_celsius, tdb_celsius, v_ms, d_m, options.emissivity)
    };
    Temperature::from_celsius(tr_celsius)
}

/// Calculate mean radiant temperature using ISO 7726:1998 method
fn mean_radiant_temperature_iso(tg: f64, tdb: f64, v: f64, d: f64, emissivity: f64) -> f64 {
    let tg_k = tg + C_TO_K;
    let tdb_k = tdb + C_TO_K;

    // Heat transfer coefficients (ISO 7726:1998)
    // Natural convection: h_n = 1.4 * (ΔT/d)^0.25
    // - 1.4 W/(m²·K): natural convection coefficient
    // - 0.25: exponent for natural convection correlation
    let h_n = 1.4 * pow(abs(tg_k - tdb_k) / d, 0.25); // natural convection

    // Forced convection: h_f = 6.3 * v^0.6 / d^0.4
    // - 6.3 W/(m²·K): forced convection coefficient
    // - 0.6: velocity exponent
    // - 0.4: diameter exponent
    let h_f = 6.3 * pow(v, 0.6) / pow(d, 0.4); // forced convection

    // Use maximum of the two. Python (`utilities.py:450`) is `np.maximum(h_f, h_n)`,
    // which propagates a NaN from either coefficient — `h_n` is NaN whenever `tg` or
    // `tdb` is, and `h_f` whenever `v` is.
    let h = np_maximum(h_f, h_n);

    pow(
        pow(tg_k, 4.0) + h * (tg_k - tdb_k) / (emissivity * 5.67e-8),
        0.25,
    ) - C_TO_K
}

/// Calculate mean radiant temperature using Mixed Convection method
fn mean_radiant_temperature_mixed(tg: f64, tdb: f64, v: f64, d: f64, emissivity: f64) -> f64 {
    // Check diameter validity
    if !(0.04..=0.15).contains(&d) {
        return f64::NAN;
    }

    const MU: f64 = 0.0000181; // Pa·s
    const K_AIR: f64 = 0.02662; // W/(m·K)
    const BETA: f64 = 0.0034; // 1/K
    const NU: f64 = 0.0000148; // m²/s
    const ALPHA: f64 = 0.00002591; // m²/s

    let pr = CP_AIR * MU / K_AIR; // Prandtl number

    let o = 5.67e-8; // Stefan-Boltzmann constant
    let n = 1.27 * d + 0.57;

    let ra = G * BETA * abs(tg - tdb) * pow(d, 3.0) / NU / ALPHA;
    let re = v * d / NU;

    let nu_natural =
        2.0 + (0.589 * pow(ra, 0.25)) / pow(1.0 + pow(0.469 / pr, 9.0 / 16.0), 4.0 / 9.0);

    let nu_forced = 2.0 + (0.4 * pow(re, 0.5) + 0.06 * pow(re, 2.0 / 3.0)) * pow(pr, 0.4);

    let nu_combined = pow(pow(nu_forced, n) + pow(nu_natural, n), 1.0 / n);

    pow(
        pow(tg + C_TO_K, 4.0) - ((nu_combined * K_AIR / d) * (-tg + tdb)) / emissivity / o,
        0.25,
    ) - C_TO_K
}

/// The comfort inputs to [`operative_temperature`]: pythermalcomfort requires all
/// three (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperativeTemperatureInputs {
    /// Air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Air speed
    pub v: Speed,
}

/// Optional parameters for [`operative_temperature`], with pythermalcomfort's
/// defaults.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OperativeTemperatureOptions {
    /// If true, use the ASHRAE method; if false, the ISO method.
    ///
    /// Mirrors pythermalcomfort's `standard` string ("ISO" / "ASHRAE"), whose default is
    /// "ISO" — which is `false` here, and so happens to coincide with `bool`'s own
    /// default. Derived rather than written out for that reason; if upstream ever
    /// defaults to ASHRAE, this needs a manual impl again.
    pub use_ashrae: bool,
}

/// Calculate operative temperature
///
/// # Returns
///
/// Operative temperature
pub fn operative_temperature(
    inputs: OperativeTemperatureInputs,
    options: OperativeTemperatureOptions,
) -> Temperature {
    let OperativeTemperatureInputs { tdb, tr, v } = inputs;
    Temperature::from_celsius(operative_temperature_celsius(
        tdb.as_celsius(),
        tr.as_celsius(),
        v.as_meters_per_second(),
        options.use_ashrae,
    ))
}

/// Plain-`f64` core of [`operative_temperature`], in °C throughout.
///
/// In-crate callers must use this rather than [`operative_temperature`]. `Temperature`
/// stores kelvin, so `from_celsius(x).as_celsius()` is not the identity -- 24.1 comes
/// back as 24.100000000000023. That is harmless for a reported value and *not* harmless
/// where the result feeds a comparison against a band edge: it made `adaptive_ashrae` and
/// `adaptive_en` disagree with pythermalcomfort on `acceptability_*` for an operative
/// temperature sitting exactly on a comfort bound, the same mechanism as the JOS3
/// ISO-7730-limit bug fixed in 28adfa8.
pub(crate) fn operative_temperature_celsius(
    tdb_celsius: f64,
    tr_celsius: f64,
    v_ms: f64,
    use_ashrae: bool,
) -> f64 {
    if use_ashrae {
        // ASHRAE 55 method with speed-dependent weighting factor
        // Thresholds and weights:
        // v < 0.2 m/s: a = 0.5 (equal weighting of air and radiant temp)
        // 0.2 ≤ v < 0.6 m/s: a = 0.6 (slightly favor air temperature)
        // v ≥ 0.6 m/s: a = 0.7 (favor air temperature more at higher speeds)
        let a = if v_ms < 0.2 {
            0.5
        } else if v_ms < 0.6 {
            0.6
        } else {
            0.7
        };
        a * tdb_celsius + (1.0 - a) * tr_celsius
    } else {
        // ISO 7730 method
        // 10.0 = velocity scaling factor for convective heat transfer
        (tdb_celsius * sqrt(10.0 * v_ms) + tr_celsius) / (1.0 + sqrt(10.0 * v_ms))
    }
}
/// The inputs to [`f_svv`]: `utilities.py:660` requires all three (no default), and they
/// are three adjacent same-typed [`Length`]s — precisely the swap hazard this pattern
/// exists for.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FSvvInputs {
    /// Width of the window
    pub width: Length,
    /// Height of the window
    pub height: Length,
    /// Distance between the occupant and the window
    pub distance: Length,
}

/// Calculate sky-vault view fraction
///
/// Calculates the fraction of the sky visible through a window.
///
/// # Returns
///
/// Sky-vault view fraction (0-1)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::{f_svv, FSvvInputs};
/// use thermalcomfort::Length;
///
/// let svv = f_svv(FSvvInputs {
///     width: Length::from_meters(2.0),
///     height: Length::from_meters(1.5),
///     distance: Length::from_meters(3.0),
/// });
/// assert!(svv > 0.0 && svv <= 1.0);
/// ```
pub fn f_svv(inputs: FSvvInputs) -> f64 {
    let FSvvInputs {
        width,
        height,
        distance,
    } = inputs;
    let w = width.as_meters();
    let h = height.as_meters();
    let d = distance.as_meters();
    let angle_h = libm::atan(h / (2.0 * d));
    let angle_w = libm::atan(w / (2.0 * d));

    // Convert radians to degrees and calculate fraction
    let degrees_h = angle_h * 180.0 / core::f64::consts::PI;
    let degrees_w = angle_w * 180.0 / core::f64::consts::PI;

    (degrees_h * degrees_w) / 16200.0
}

/// Transpose the solar altitude and solar azimuth angles, in plain degrees.
///
/// The in-crate entry point for [`crate::models::solar_gain`], which already holds its
/// `sharp`/`sol_altitude` as unwrapped `f64` degrees (see [`solar_gain`](crate::models::solar_gain::solar_gain)'s
/// "Convert once here, then calculate in plain f64 throughout"). Going through the
/// typed [`transpose_sharp_altitude`] wrapper there would mean re-wrapping those `f64`s
/// into an [`measurements::Angle`] just to call it and unwrap again — exactly the
/// round-trip anti-pattern that cost a 2 °C bug elsewhere in this crate (28adfa8).
///
/// # Returns
///
/// Tuple of (transposed sharp, transposed altitude) in degrees, each rounded to 3
/// decimals to match pythermalcomfort.
pub(crate) fn transpose_sharp_altitude_degrees(sharp: f64, altitude: f64) -> (f64, f64) {
    let to_rad = core::f64::consts::PI / 180.0;
    let to_deg = 180.0 / core::f64::consts::PI;

    let altitude_new =
        libm::asin(libm::sin(libm::fabs(sharp - 90.0) * to_rad) * libm::cos(altitude * to_rad))
            * to_deg;
    let sharp_new =
        libm::atan(libm::sin(sharp * to_rad) * libm::tan((90.0 - altitude) * to_rad)) * to_deg;

    // utilities.py:490's `transpose_sharp_altitude` is `@njit(cache=True)`-decorated.
    // Numba compiles `round(x, n)` to its own multiply/rint/divide sequence
    // (numba/cpython/builtins.py:262-278), which is numpy's ties-to-even rule, not
    // CPython's exact-decimal builtin — confirmed by reading that source and by probing
    // the compiled function directly: over 2,000,000 engineered tie-adjacent values it
    // agreed with `np.round` on all of them and disagreed with the plain-Python `round`
    // builtin on 5. So despite `sharp`/`altitude` being ordinary `float` parameters, the
    // jit context makes this numpy's rule.
    (round_to(sharp_new, 3), round_to(altitude_new, 3))
}

/// Transpose the solar altitude and solar azimuth angles
///
/// Used by [`fn@crate::models::solar_gain`] to reuse the standing projected-area table for
/// a supine occupant, by rotating the sun's position into the body's frame.
///
/// A thin newtype wrapper over `transpose_sharp_altitude_degrees`; see that function's
/// doc comment for why the crate keeps a plain-`f64` core rather than only this typed
/// entry point.
///
/// # Returns
///
/// Tuple of (transposed sharp, transposed altitude), each rounded to 3 decimals to
/// match pythermalcomfort.
///
/// The arguments are in the order the name gives, `sharp` then `altitude`, which is the
/// reverse of [`SolarGainInputs`](crate::models::solar_gain::SolarGainInputs), where
/// `sol_altitude` precedes `sharp`. Both are `Angle`s, so a swap compiles: name the
/// values rather than passing positionally from that struct.
///
/// # Examples
///
/// ```
/// use thermalcomfort::environment::transpose_sharp_altitude;
/// use thermalcomfort::Angle;
///
/// let (sharp, altitude) = transpose_sharp_altitude(Angle::from_degrees(0.0), Angle::from_degrees(0.0));
/// assert_eq!((sharp.as_degrees(), altitude.as_degrees()), (0.0, 90.0));
/// ```
pub fn transpose_sharp_altitude(sharp: Angle, altitude: Angle) -> (Angle, Angle) {
    let (sharp_new, altitude_new) =
        transpose_sharp_altitude_degrees(sharp.as_degrees(), altitude.as_degrees());
    (
        Angle::from_degrees(sharp_new),
        Angle::from_degrees(altitude_new),
    )
}

/// The inputs to [`scale_wind_speed_log`]: pythermalcomfort requires both (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleWindSpeedLogInputs {
    /// Wind speed measured at the reference height `z1`
    pub v_z1: Speed,
    /// Height the wind speed is wanted at
    pub z2: Length,
}

/// Optional parameters for [`scale_wind_speed_log`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleWindSpeedLogOptions {
    /// Reference height of the measurement. Defaults to 10 m.
    pub z1: Length,
    /// Surface roughness length. Defaults to 0.01 m (open terrain); 0.03 m pasture,
    /// 0.1 m crops, 0.4 m towns and forests, 1 m urban, 1.6 m large cities.
    pub z0: Length,
    /// Zero-plane displacement height, the height at which the wind effectively drops
    /// to zero among obstacles. Defaults to 0 m.
    pub d: Length,
    /// Round the result to 0.01 m/s, as upstream does. Defaults to `true`.
    pub round_output: bool,
}

impl Default for ScaleWindSpeedLogOptions {
    fn default() -> Self {
        Self {
            z1: Length::from_meters(10.0),
            z0: Length::from_meters(0.01),
            d: Length::from_meters(0.0),
            round_output: true,
        }
    }
}

/// A geometry the logarithmic wind profile is not defined for.
///
/// Upstream raises `ValueError` for each; the variants are its checks, in its order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleWindSpeedLogError {
    /// `v_z1 < 0`
    NegativeWindSpeed,
    /// `z0 <= 0`
    NonPositiveRoughness,
    /// `d < 0`
    NegativeDisplacement,
    /// `z2 <= 0`
    NonPositiveTargetHeight,
    /// `z1 <= 0`
    NonPositiveReferenceHeight,
    /// `z2 <= z0`
    TargetNotAboveRoughness,
    /// `z1 <= z0`
    ReferenceNotAboveRoughness,
    /// `z1 - d <= z0`
    ReferenceMinusDisplacementNotAboveRoughness,
    /// `z2 - d <= z0`
    TargetMinusDisplacementNotAboveRoughness,
}

impl core::fmt::Display for ScaleWindSpeedLogError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NegativeWindSpeed => "wind speed (v_z1) must be non-negative",
            Self::NonPositiveRoughness => "surface roughness length (z0) must be positive",
            Self::NegativeDisplacement => "zero-plane displacement height (d) must be >= 0",
            Self::NonPositiveTargetHeight => "target height (z2) must be positive",
            Self::NonPositiveReferenceHeight => "reference height (z1) must be positive",
            Self::TargetNotAboveRoughness => {
                "target height (z2) must be greater than surface roughness (z0)"
            }
            Self::ReferenceNotAboveRoughness => {
                "reference height (z1) must be greater than surface roughness (z0)"
            }
            Self::ReferenceMinusDisplacementNotAboveRoughness => {
                "reference height minus displacement (z1 - d) must be > z0"
            }
            Self::TargetMinusDisplacementNotAboveRoughness => {
                "target height minus displacement (z2 - d) must be > z0"
            }
        })
    }
}

impl core::error::Error for ScaleWindSpeedLogError {}

/// Scale a wind speed from its reference height to another height with the logarithmic
/// wind profile (Oke 1987).
///
/// `v(z2) = v(z1) · ln((z2 - d) / z0) / ln((z1 - d) / z0)`, which assumes neutral
/// atmospheric stability and holds from a few metres to about 100 m above ground. It
/// does not account for obstacles or complex terrain.
///
/// # Errors
///
/// Returns [`ScaleWindSpeedLogError`] for the geometries upstream rejects: a negative
/// wind speed, a non-positive roughness length or height, a negative displacement, or a
/// height that, less the displacement, does not clear the roughness length.
///
/// # Examples
///
/// ```
/// use thermalcomfort::environment::{scale_wind_speed_log, ScaleWindSpeedLogInputs};
/// use thermalcomfort::{Length, Speed};
///
/// // A 10 m anemometer reading, brought down to 2 m over open terrain
/// let v_2m = scale_wind_speed_log(
///     ScaleWindSpeedLogInputs {
///         v_z1: Speed::from_meters_per_second(5.0),
///         z2: Length::from_meters(2.0),
///     },
///     Default::default(),
/// )
/// .expect("2 m is above the roughness length");
/// assert_eq!(v_2m.as_meters_per_second(), 3.84);
/// ```
///
/// # References
///
/// - Oke, T. R. (1987). Boundary Layer Climates.
pub fn scale_wind_speed_log(
    inputs: ScaleWindSpeedLogInputs,
    options: ScaleWindSpeedLogOptions,
) -> Result<Speed, ScaleWindSpeedLogError> {
    let v_z1 = inputs.v_z1.as_meters_per_second();
    let z2 = inputs.z2.as_meters();
    let z1 = options.z1.as_meters();
    let z0 = options.z0.as_meters();
    let d = options.d.as_meters();

    // Upstream's checks, in upstream's order; NaN passes every one, as it does there.
    if v_z1 < 0.0 {
        return Err(ScaleWindSpeedLogError::NegativeWindSpeed);
    }
    if z0 <= 0.0 {
        return Err(ScaleWindSpeedLogError::NonPositiveRoughness);
    }
    if d < 0.0 {
        return Err(ScaleWindSpeedLogError::NegativeDisplacement);
    }
    if z2 <= 0.0 {
        return Err(ScaleWindSpeedLogError::NonPositiveTargetHeight);
    }
    if z1 <= 0.0 {
        return Err(ScaleWindSpeedLogError::NonPositiveReferenceHeight);
    }
    if z2 <= z0 {
        return Err(ScaleWindSpeedLogError::TargetNotAboveRoughness);
    }
    if z1 <= z0 {
        return Err(ScaleWindSpeedLogError::ReferenceNotAboveRoughness);
    }
    if z1 - d <= z0 {
        return Err(ScaleWindSpeedLogError::ReferenceMinusDisplacementNotAboveRoughness);
    }
    if z2 - d <= z0 {
        return Err(ScaleWindSpeedLogError::TargetMinusDisplacementNotAboveRoughness);
    }

    let mut v_z2 = v_z1 * log((z2 - d) / z0) / log((z1 - d) / z0);
    if options.round_output {
        v_z2 = round_to(v_z2, 2);
    }
    Ok(Speed::from_meters_per_second(v_z2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v_relative() {
        let v1 = v_relative(
            Speed::from_meters_per_second(0.1),
            MetabolicRate::from_met(1.0),
        );
        assert_eq!(v1.as_meters_per_second(), 0.1);

        let v2 = v_relative(
            Speed::from_meters_per_second(0.1),
            MetabolicRate::from_met(1.4),
        );
        assert!((v2.as_meters_per_second() - 0.22).abs() < 0.001);

        let v3 = v_relative(
            Speed::from_meters_per_second(0.15),
            MetabolicRate::from_met(2.0),
        );
        assert!((v3.as_meters_per_second() - 0.45).abs() < 0.001);
    }
    #[test]
    fn test_mean_radiant_temperature_defaults() {
        // Python defaults d=0.15, emissivity=0.95, standard="Mixed Convection".
        let result = mean_radiant_temperature(
            MeanRadiantTemperatureInputs {
                tg: Temperature::from_celsius(30.0),
                tdb: Temperature::from_celsius(25.0),
                v: Speed::from_meters_per_second(0.3),
            },
            Default::default(),
        );
        let explicit = mean_radiant_temperature(
            MeanRadiantTemperatureInputs {
                tg: Temperature::from_celsius(30.0),
                tdb: Temperature::from_celsius(25.0),
                v: Speed::from_meters_per_second(0.3),
            },
            MeanRadiantTemperatureOptions {
                d: Length::from_meters(0.15),
                emissivity: 0.95,
                use_iso: false,
            },
        );
        assert_eq!(result.as_celsius(), explicit.as_celsius());
    }
    #[test]
    fn test_operative_temperature() {
        // ISO method (default)
        let t_op = operative_temperature(
            OperativeTemperatureInputs {
                tdb: Temperature::from_celsius(25.0),
                tr: Temperature::from_celsius(25.0),
                v: Speed::from_meters_per_second(0.1),
            },
            Default::default(),
        );
        assert!((t_op.as_celsius() - 25.0).abs() < 0.01);

        // ASHRAE method
        let t_op2 = operative_temperature(
            OperativeTemperatureInputs {
                tdb: Temperature::from_celsius(22.0),
                tr: Temperature::from_celsius(26.0),
                v: Speed::from_meters_per_second(0.1),
            },
            OperativeTemperatureOptions { use_ashrae: true },
        );
        assert!(t_op2.as_celsius() > 22.0 && t_op2.as_celsius() < 26.0);
    }
    #[test]
    fn test_f_svv() {
        let svv = f_svv(FSvvInputs {
            width: Length::from_meters(2.0),
            height: Length::from_meters(1.5),
            distance: Length::from_meters(3.0),
        });
        assert!(svv > 0.0 && svv <= 1.0);
    }
    #[test]
    fn test_transpose_sharp_altitude() {
        let (sharp_t, alt_t) =
            transpose_sharp_altitude(Angle::from_degrees(30.0), Angle::from_degrees(45.0));
        assert!(sharp_t.as_degrees() > 0.0);
        assert!(alt_t.as_degrees() > 0.0);
    }
    #[test]
    fn test_transpose_sharp_altitude_degrees_matches_typed_wrapper() {
        let (sharp_deg, alt_deg) = transpose_sharp_altitude_degrees(30.0, 45.0);
        let (sharp_t, alt_t) =
            transpose_sharp_altitude(Angle::from_degrees(30.0), Angle::from_degrees(45.0));
        assert!((sharp_t.as_degrees() - sharp_deg).abs() < 1e-9);
        assert!((alt_t.as_degrees() - alt_deg).abs() < 1e-9);
    }

    #[test]
    fn scale_wind_speed_log_matches_the_profile() {
        // Upstream's own cases, v(z1) · ln((z2 - d)/z0) / ln((z1 - d)/z0), unrounded.
        let run = |v: f64, z2: f64, z1: f64, z0: f64, d: f64| {
            scale_wind_speed_log(
                ScaleWindSpeedLogInputs {
                    v_z1: Speed::from_meters_per_second(v),
                    z2: Length::from_meters(z2),
                },
                ScaleWindSpeedLogOptions {
                    z1: Length::from_meters(z1),
                    z0: Length::from_meters(z0),
                    d: Length::from_meters(d),
                    round_output: false,
                },
            )
            .map(|s| s.as_meters_per_second())
        };
        let expected =
            |v: f64, z2: f64, z1: f64, z0: f64, d: f64| v * log((z2 - d) / z0) / log((z1 - d) / z0);
        assert_eq!(
            run(5.0, 2.0, 10.0, 0.01, 0.0),
            Ok(expected(5.0, 2.0, 10.0, 0.01, 0.0))
        );
        assert_eq!(
            run(5.0, 2.0, 10.0, 0.1, 0.5),
            Ok(expected(5.0, 2.0, 10.0, 0.1, 0.5))
        );
        // Going up from the reference height increases the speed; down decreases it.
        assert!(run(5.0, 90.0, 10.0, 0.1, 0.0).unwrap() > 5.0);
        assert!(run(5.0, 2.0, 10.0, 0.1, 0.0).unwrap() < 5.0);
    }

    #[test]
    fn scale_wind_speed_log_rejects_what_upstream_rejects() {
        use ScaleWindSpeedLogError::*;
        let run = |v: f64, z2: f64, z1: f64, z0: f64, d: f64| {
            scale_wind_speed_log(
                ScaleWindSpeedLogInputs {
                    v_z1: Speed::from_meters_per_second(v),
                    z2: Length::from_meters(z2),
                },
                ScaleWindSpeedLogOptions {
                    z1: Length::from_meters(z1),
                    z0: Length::from_meters(z0),
                    d: Length::from_meters(d),
                    round_output: false,
                },
            )
        };
        assert_eq!(run(-1.0, 2.0, 10.0, 0.01, 0.0), Err(NegativeWindSpeed));
        assert_eq!(
            run(5.0, -2.0, 10.0, 0.01, 0.0),
            Err(NonPositiveTargetHeight)
        );
        assert_eq!(run(5.0, 2.0, 10.0, 0.0, 0.0), Err(NonPositiveRoughness));
        assert_eq!(run(5.0, 2.0, 10.0, -0.1, 0.0), Err(NonPositiveRoughness));
        assert_eq!(run(5.0, 2.0, 10.0, 0.01, -0.5), Err(NegativeDisplacement));
        assert_eq!(
            run(5.0, 2.0, -10.0, 0.01, 0.0),
            Err(NonPositiveReferenceHeight)
        );
        assert_eq!(
            run(5.0, 0.01, 10.0, 0.01, 0.0),
            Err(TargetNotAboveRoughness)
        );
        assert_eq!(
            run(5.0, 2.0, 0.01, 0.01, 0.0),
            Err(ReferenceNotAboveRoughness)
        );
        assert_eq!(
            run(5.0, 2.0, 1.0, 0.01, 1.0),
            Err(ReferenceMinusDisplacementNotAboveRoughness)
        );
        assert_eq!(
            run(5.0, 2.0, 10.0, 0.01, 2.0),
            Err(TargetMinusDisplacementNotAboveRoughness)
        );
    }
}
