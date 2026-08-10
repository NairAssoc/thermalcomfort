//! PMV (Predicted Mean Vote) and PPD (Predicted Percentage Dissatisfied) models
//!
//! Implementation of thermal comfort models according to ISO 7730 and ASHRAE 55 standards

use crate::constants::*;
use crate::utilities::{Ashrae55Model, round_to, valid_range};
use crate::{ClothingInsulation, MetabolicRate};
use libm::{exp, fabs as abs, fmax, pow, sqrt};
use measurements::{Humidity, Speed, Temperature};

/// Result of PMV/PPD calculation
#[derive(Debug, Clone, Copy)]
pub struct PmvPpdResult {
    /// Predicted Mean Vote (PMV)
    pub pmv: f64,
    /// Predicted Percentage of Dissatisfied (PPD) [%]
    pub ppd: f64,
    /// Thermal sensation vote category.
    ///
    /// `None` when the PMV is not available (NaN) because the inputs fell outside the
    /// model's applicability limits, matching Python's `nan` for this field.
    pub tsv: Option<ThermalSensation>,
    /// ASHRAE 55:2023 compliance: `Some(true)` when -0.5 < PMV < 0.5, `Some(false)`
    /// otherwise. Only populated by [`pmv_ppd_ashrae`]; ISO and other variants
    /// leave this as `None` because compliance with the ASHRAE comfort criterion
    /// is meaningful only for the ASHRAE model.
    pub compliance: Option<bool>,
}

/// Thermal sensation categories based on PMV value
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalSensation {
    Cold,
    Cool,
    SlightlyCool,
    Neutral,
    SlightlyWarm,
    Warm,
    Hot,
}

impl ThermalSensation {
    /// Map a PMV value to a thermal sensation category, or `None` when the PMV is not
    /// available (NaN). Python yields `nan` for `tsv` in that case rather than a band.
    ///
    /// There is deliberately no infallible `from_pmv`: mapping NaN to a band reports a
    /// comfort category for a calculation that did not produce one, which is the defect
    /// the `Option` on [`PmvPpdResult::tsv`] exists to fix.
    pub fn from_pmv_opt(pmv: f64) -> Option<Self> {
        Self::from_pmv_banded(pmv, false)
    }

    /// Categorise a PMV using ASHRAE's right-closed bands.
    ///
    /// ISO 7730 maps with `right=False` and ASHRAE 55 with the default `right=True`,
    /// so a PMV sitting exactly on an edge lands in different bands. Since PMV is
    /// rounded to two decimals first, exact edges like -0.50 are common: there ISO
    /// gives Neutral and ASHRAE gives Slightly Cool.
    pub fn from_pmv_opt_ashrae(pmv: f64) -> Option<Self> {
        Self::from_pmv_banded(pmv, true)
    }

    fn from_pmv_banded(pmv: f64, right_closed: bool) -> Option<Self> {
        if pmv.is_nan() {
            return None;
        }

        let below = |edge: f64| {
            if right_closed {
                pmv <= edge
            } else {
                pmv < edge
            }
        };

        Some(if below(-2.5) {
            ThermalSensation::Cold
        } else if below(-1.5) {
            ThermalSensation::Cool
        } else if below(-0.5) {
            ThermalSensation::SlightlyCool
        } else if below(0.5) {
            ThermalSensation::Neutral
        } else if below(1.5) {
            ThermalSensation::SlightlyWarm
        } else if below(2.5) {
            ThermalSensation::Warm
        } else {
            ThermalSensation::Hot
        })
    }

    /// String form matching pythermalcomfort's `tsv` field exactly.
    pub fn as_str(&self) -> &'static str {
        match self {
            ThermalSensation::Cold => "Cold",
            ThermalSensation::Cool => "Cool",
            ThermalSensation::SlightlyCool => "Slightly Cool",
            ThermalSensation::Neutral => "Neutral",
            ThermalSensation::SlightlyWarm => "Slightly Warm",
            ThermalSensation::Warm => "Warm",
            ThermalSensation::Hot => "Hot",
        }
    }
}

/// The comfort inputs shared by [`pmv_ppd_iso`] and [`pmv_ppd_ashrae`]: pythermalcomfort
/// requires all six for both functions (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvPpdInputs {
    /// Dry bulb air temperature
    pub dry_bulb_temp: Temperature,
    /// Mean radiant temperature
    pub mean_radiant_temp: Temperature,
    /// Relative air speed
    pub relative_air_speed: Speed,
    /// Relative humidity
    pub relative_humidity: Humidity,
    /// Metabolic rate
    pub metabolic_rate: MetabolicRate,
    /// Clothing insulation
    pub clothing_insulation: ClothingInsulation,
}

/// Edition selector for [`pmv_ppd_iso`].
///
/// pythermalcomfort's `pmv_ppd_iso` takes a `model: str` parameter (default
/// `"7730-2025"`, legal values `"7730-2005"` and `"7730-2025"`). The PMV/PPD formulae
/// are unchanged between the two editions, so both variants are behaviourally identical
/// TODAY — but the selector is still ported (rather than collapsed to nothing) because
/// it is a documented, user-facing part of upstream's signature. Matching on this enum
/// is exhaustive, so a future edition that actually diverges becomes a compile error
/// everywhere it needs handling, not a silent no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Iso7730Model {
    /// ISO 7730:2005
    Iso77302005,
    /// ISO 7730:2025 (the current edition, and upstream's default)
    #[default]
    Iso77302025,
}

/// PMV/PPD calculation options for [`pmv_ppd_iso`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvPpdIsoOptions {
    /// External work, default 0 met
    pub wme: MetabolicRate,
    /// Standard edition. Behaviourally identical to the other value today, see
    /// [`Iso7730Model`].
    pub model: Iso7730Model,
    /// Limit inputs to standard compliance ranges
    pub limit_inputs: bool,
    /// Round output values
    pub round_output: bool,
}

impl Default for PmvPpdIsoOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            model: Iso7730Model::default(),
            limit_inputs: true,
            round_output: true,
        }
    }
}

/// PMV/PPD calculation options for [`pmv_ppd_ashrae`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvPpdAshraeOptions {
    /// External work, default 0 met
    pub wme: MetabolicRate,
    /// Standard edition. Behaviourally identical to the only other legal value today
    /// (there is only one), see [`Ashrae55Model`].
    pub model: Ashrae55Model,
    /// Limit inputs to standard compliance ranges
    pub limit_inputs: bool,
    /// Whether the occupant has control over the airspeed.
    ///
    /// When `true` (upstream's default), ASHRAE 55 imposes no airspeed limit beyond the
    /// flat Table 7.3.4 range (0-2 m/s). When `false`, `§7.2.1.2`'s cross-variable rules
    /// additionally NaN the result whenever elevated air speed is combined with low
    /// clothing (clo < 0.7) and low activity (met < 1.3) — either unconditionally above
    /// 0.8 m/s, or above a `to`-dependent limit inside/below the comfort band. See
    /// [`check_ashrae55_compliance`].
    pub airspeed_control: bool,
    /// Round output values
    pub round_output: bool,
}

impl Default for PmvPpdAshraeOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            model: Ashrae55Model::default(),
            limit_inputs: true,
            airspeed_control: true,
            round_output: true,
        }
    }
}

/// ASHRAE 55-2023 input applicability check (Table 7.3.4 single-variable limits plus
/// the `§7.2.1.2` `airspeed_control` cross-variable rules).
///
/// Ported from pythermalcomfort's `_check_ashrae55_compliance` in `utilities.py`.
/// Returns `(tdb, tr, v, met, clo)`, each either the input value (if valid) or `NaN`;
/// the caller treats any `NaN` among them as "outside ASHRAE 55's applicability".
///
/// `tdb`, `tr`, and `v` here are the *raw* (pre cooling-effect-adjustment) values —
/// matching upstream, which runs this check against the caller's original inputs, not
/// the Appendix H3 cooling-effect-adjusted ones used for the PMV calculation itself.
fn check_ashrae55_compliance(
    tdb: f64,
    tr: f64,
    v: f64,
    met: f64,
    clo: f64,
    airspeed_control: bool,
) -> (f64, f64, f64, f64, f64) {
    let tdb_valid = valid_range(tdb, 10.0, 40.0);
    let tr_valid = valid_range(tr, 10.0, 40.0);
    let mut v_valid = valid_range(v, 0.0, 2.0);

    if !airspeed_control {
        // Elevated air speed with low clothing and low activity: unconditionally
        // outside the standard above 0.8 m/s.
        if v > 0.8 && clo < 0.7 && met < 1.3 {
            v_valid = f64::NAN;
        }

        // Operative temperature (ISO formula, upstream's default `standard="ISO"`):
        // to = (tdb*sqrt(10v) + tr) / (1 + sqrt(10v))
        let sqrt_10v = sqrt(10.0 * v);
        let to = (tdb * sqrt_10v + tr) / (1.0 + sqrt_10v);
        let v_limit = 50.49 - 4.4047 * to + 0.096425 * to * to;

        // Inside the comfort band (23-25.5°C operative), a `to`-dependent airspeed
        // limit applies.
        if to > 23.0 && to < 25.5 && v > v_limit && clo < 0.7 && met < 1.3 {
            v_valid = f64::NAN;
        }

        // Below the comfort band, the flat 0.2 m/s limit applies.
        if to <= 23.0 && v > 0.2 && clo < 0.7 && met < 1.3 {
            v_valid = f64::NAN;
        }
    }

    let met_valid = valid_range(met, 1.0, 4.0);
    let clo_valid = valid_range(clo, 0.0, 1.5);

    (tdb_valid, tr_valid, v_valid, met_valid, clo_valid)
}

/// Calculate PMV and PPD according to ISO 7730
///
/// Returns the Predicted Mean Vote (PMV) and Predicted Percentage of Dissatisfied (PPD)
/// calculated in accordance with ISO 7730. The ISO uses the same formulation of PMV
/// as published by Fanger (1970).
///
/// # Returns
///
/// `PmvPpdResult` containing PMV, PPD, and thermal sensation category
///
/// # Standard Compliance Limits (ISO 7730 Clause 4)
///
/// When `limit_inputs` is true:
/// - 10 < tdb [°C] < 30
/// - 10 < tr [°C] < 40
/// - 0 < vr [m/s] < 1
/// - 0.8 < met < 4
/// - 0 < clo < 2
/// - 0 < pa [Pa] < 2700 (water vapour partial pressure, derived from tdb and rh)
/// - -2 < PMV < 2
///
/// # Example
///
/// ```
/// use thermalcomfort::models::pmv::{pmv_ppd_iso, PmvPpdInputs, PmvPpdIsoOptions};
/// use thermalcomfort::utilities::v_relative;
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let tdb = Temperature::from_celsius(25.0);
/// let tr = Temperature::from_celsius(25.0);
/// let rh = Humidity::from_percent(50.0);
/// let v = Speed::from_meters_per_second(0.1);
/// let met = MetabolicRate::from_met(1.4);
/// let clo = ClothingInsulation::from_clo(0.5);
///
/// let vr = v_relative(v, met);
/// let result = pmv_ppd_iso(
///     PmvPpdInputs {
///         dry_bulb_temp: tdb,
///         mean_radiant_temp: tr,
///         relative_air_speed: vr,
///         relative_humidity: rh,
///         metabolic_rate: met,
///         clothing_insulation: clo,
///     },
///     Default::default(),
/// );
/// // result.pmv ≈ 0.17, result.ppd ≈ 5.6
/// ```
pub fn pmv_ppd_iso(inputs: PmvPpdInputs, options: PmvPpdIsoOptions) -> PmvPpdResult {
    let PmvPpdInputs {
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
    } = inputs;
    pmv_ppd_iso_celsius(
        dry_bulb_temp.as_celsius(),
        mean_radiant_temp.as_celsius(),
        relative_air_speed.as_meters_per_second(),
        relative_humidity.as_percent(),
        metabolic_rate.as_met(),
        clothing_insulation.as_clo(),
        options,
    )
}

/// ISO 7730 PMV/PPD from plain `f64` inputs, in Celsius and SI.
///
/// The public [`pmv_ppd_iso`] is a thin newtype wrapper over this. Callers *inside* this
/// crate that already hold plain `f64` values must use this entry point rather than
/// wrapping them up again, because `Temperature` stores kelvin and
/// `Temperature::from_celsius(x).as_celsius()` is not the identity — it loses the last
/// ULP. That is normally harmless, but `limit_inputs` compares against inclusive bounds,
/// so a value a single ULP outside a limit round-trips back onto it and silently passes a
/// check upstream fails.
///
/// JOS3 is where this bit: its operative-temperature search deliberately walks `to` onto
/// the 30 °C ISO limit and depends on crossing it to trigger a NaN-and-retry branch. Going
/// through the newtype meant the crossing never registered, the retry never ran, and the
/// subject's set-point temperatures were left about 2 °C wrong for the life of the model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pmv_ppd_iso_celsius(
    dry_bulb_celsius: f64,
    radiant_celsius: f64,
    air_speed: f64,
    rh_percent: f64,
    met: f64,
    clo: f64,
    options: PmvPpdIsoOptions,
) -> PmvPpdResult {
    let PmvPpdIsoOptions {
        wme,
        model,
        limit_inputs,
        round_output,
    } = options;

    // Exhaustive match: both editions are formula-identical today (see
    // `Iso7730Model`), but this stops a future divergent edition from being silently
    // ignored.
    match model {
        Iso7730Model::Iso77302005 | Iso7730Model::Iso77302025 => {}
    }

    // Check standard compliance if requested
    if limit_inputs {
        let dry_bulb_valid = valid_range(dry_bulb_celsius, 10.0, 30.0);
        let radiant_valid = valid_range(radiant_celsius, 10.0, 40.0);
        let speed_valid = valid_range(air_speed, 0.0, 1.0);
        let metabolic_valid = valid_range(met, 0.8, 4.0);
        let clothing_valid = valid_range(clo, 0.0, 2.0);
        // ISO 7730 Clause 4 also bounds water vapour partial pressure to 0-2700 Pa.
        // e.g. tdb=30, rh=100 gives pa ~4243 Pa, outside the standard's applicability.
        let pa = rh_percent * 10.0 * exp(16.6536 - 4030.183 / (dry_bulb_celsius + 235.0));
        let pa_valid = valid_range(pa, 0.0, 2700.0);

        if dry_bulb_valid.is_nan()
            || radiant_valid.is_nan()
            || speed_valid.is_nan()
            || metabolic_valid.is_nan()
            || clothing_valid.is_nan()
            || pa_valid.is_nan()
        {
            return PmvPpdResult {
                pmv: f64::NAN,
                ppd: f64::NAN,
                tsv: None,
                compliance: None,
            };
        }
    }

    // Calculate PMV using optimized algorithm
    let pmv = pmv_optimized(
        dry_bulb_celsius,
        radiant_celsius,
        air_speed,
        rh_percent,
        met,
        clo,
        wme.as_met(),
    );

    // Check PMV range if limiting inputs
    let pmv_final = if limit_inputs {
        let pmv_valid = valid_range(pmv, -2.0, 2.0);
        if pmv_valid.is_nan() {
            return PmvPpdResult {
                pmv: f64::NAN,
                ppd: f64::NAN,
                tsv: None,
                compliance: None,
            };
        }
        pmv_valid
    } else {
        pmv
    };

    // Calculate PPD (Predicted Percentage of Dissatisfied) from PMV
    // PPD equation from Fanger's model (ISO 7730):
    // PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)
    // Constants: 95.0 = percentage scale factor
    //           0.03353 = quartic term coefficient
    //           0.2179 = quadratic term coefficient
    let ppd = 100.0 - 95.0 * exp(-0.03353 * pow(pmv_final, 4.0) - 0.2179 * pow(pmv_final, 2.0));

    // Round if requested
    let (pmv_out, ppd_out) = if round_output {
        (round_to(pmv_final, 2), round_to(ppd, 1))
    } else {
        (pmv_final, ppd)
    };

    PmvPpdResult {
        pmv: pmv_out,
        ppd: ppd_out,
        tsv: ThermalSensation::from_pmv_opt(pmv_out),
        // ISO 7730 does not define an ASHRAE-style compliance check.
        compliance: None,
    }
}

/// Calculate PMV and PPD according to ASHRAE 55
///
/// Similar to ISO 7730 but with different applicability limits
///
/// # Returns
///
/// `PmvPpdResult` containing PMV, PPD, and thermal sensation category
///
/// # Standard Compliance Limits (ASHRAE 55-2023)
///
/// When `limit_inputs` is true:
/// - 10 < tdb [°C] < 40
/// - 10 < tr [°C] < 40
/// - 0 < vr [m/s] < 2
/// - 1.0 < met < 4
/// - 0 < clo < 1.5
///
/// When additionally `airspeed_control` is false, `§7.2.1.2`'s cross-variable rules
/// also NaN the result for elevated air speed combined with low clothing/activity; see
/// [`check_ashrae55_compliance`].
///
/// # Example
///
/// ```
/// use thermalcomfort::models::pmv::{pmv_ppd_ashrae, PmvPpdInputs, PmvPpdAshraeOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let result = pmv_ppd_ashrae(
///     PmvPpdInputs {
///         dry_bulb_temp: Temperature::from_celsius(25.0),
///         mean_radiant_temp: Temperature::from_celsius(25.0),
///         relative_air_speed: Speed::from_meters_per_second(0.1),
///         relative_humidity: Humidity::from_percent(50.0),
///         metabolic_rate: MetabolicRate::from_met(1.4),
///         clothing_insulation: ClothingInsulation::from_clo(0.5),
///     },
///     Default::default(),
/// );
/// // result.pmv ≈ 0.0, result.compliance == Some(true)
/// ```
pub fn pmv_ppd_ashrae(inputs: PmvPpdInputs, options: PmvPpdAshraeOptions) -> PmvPpdResult {
    let PmvPpdInputs {
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
    } = inputs;
    let PmvPpdAshraeOptions {
        wme,
        model,
        limit_inputs,
        airspeed_control,
        round_output,
    } = options;

    // Exhaustive match: there is only one ASHRAE 55 edition today (see
    // `Ashrae55Model`), but this stops a future edition from being silently ignored.
    match model {
        Ashrae55Model::Ashrae552023 => {}
    }

    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let air_speed = relative_air_speed.as_meters_per_second();
    let rh_percent = relative_humidity.as_percent();
    let met = metabolic_rate.as_met();
    let clo = clothing_insulation.as_clo();

    // Check ASHRAE standard compliance if requested. This runs against the *raw*
    // (pre cooling-effect-adjustment) tdb/tr/vr, matching upstream.
    if limit_inputs {
        let (tdb_valid, tr_valid, v_valid, met_valid, clo_valid) = check_ashrae55_compliance(
            dry_bulb_celsius,
            radiant_celsius,
            air_speed,
            met,
            clo,
            airspeed_control,
        );

        if tdb_valid.is_nan()
            || tr_valid.is_nan()
            || v_valid.is_nan()
            || met_valid.is_nan()
            || clo_valid.is_nan()
        {
            return PmvPpdResult {
                pmv: f64::NAN,
                ppd: f64::NAN,
                tsv: None,
                compliance: None,
            };
        }
    }

    // ASHRAE 55 Appendix H, H3: when relative air speed exceeds 0.1 m/s,
    // apply the cooling-effect correction. The cooling effect (ce) is the
    // temperature reduction at still air that yields the same SET as the
    // elevated-airspeed environment. We then compute PMV against the adjusted
    // (tdb-ce, tr-ce, 0.1 m/s) environment, matching pythermalcomfort.
    let (tdb_adj, tr_adj, vr_adj) = if air_speed > 0.1 {
        let ce = crate::models::cooling_effect::cooling_effect(
            crate::models::cooling_effect::CoolingEffectInputs {
                dry_bulb_temp,
                mean_radiant_temp,
                relative_air_speed,
                relative_humidity,
                metabolic_rate,
                clothing_insulation,
            },
            crate::models::cooling_effect::CoolingEffectOptions {
                wme,
                ..Default::default()
            },
        )
        .as_celsius();
        if ce > 0.0 {
            (dry_bulb_celsius - ce, radiant_celsius - ce, 0.1)
        } else {
            (dry_bulb_celsius, radiant_celsius, air_speed)
        }
    } else {
        (dry_bulb_celsius, radiant_celsius, air_speed)
    };

    // Calculate PMV (same algorithm as ISO) on the cooling-effect-adjusted inputs
    let pmv = pmv_optimized(tdb_adj, tr_adj, vr_adj, rh_percent, met, clo, wme.as_met());

    // Calculate PPD from PMV
    let ppd = 100.0 - 95.0 * exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0));

    // Round if requested
    let (pmv_out, ppd_out) = if round_output {
        (round_to(pmv, 2), round_to(ppd, 1))
    } else {
        (pmv, ppd)
    };

    // ASHRAE 55:2023 comfort criterion: -0.5 < PMV < 0.5.
    // pythermalcomfort evaluates this on the unrounded PMV (before round_output).
    let compliance = if pmv.is_nan() {
        None
    } else {
        Some(pmv > -0.5 && pmv < 0.5)
    };

    PmvPpdResult {
        pmv: pmv_out,
        ppd: ppd_out,
        tsv: ThermalSensation::from_pmv_opt_ashrae(pmv_out),
        compliance,
    }
}

/// Optimized PMV calculation core algorithm
///
/// This is the core PMV calculation from Fanger's model,
/// ported from the numba-optimized Python version
///
/// # Sources
/// The constants used in this calculation come from:
/// - ISO 7730:2005 - Ergonomics of the thermal environment
/// - Fanger, P.O. (1970) - Thermal Comfort
/// - ASHRAE Standard 55 - Thermal Environmental Conditions for Human Occupancy
fn pmv_optimized(tdb: f64, tr: f64, vr: f64, rh: f64, met: f64, clo: f64, wme: f64) -> f64 {
    // Calculate partial vapor pressure using Antoine equation
    // Constants: 16.6536, 4030.183, 235.0 are from the simplified Antoine equation
    // for water vapor pressure calculation
    let pa = rh * 10.0 * exp(16.6536 - 4030.183 / (tdb + 235.0));

    // Thermal insulation of clothing [m²K/W]
    // 0.155 = 1 clo in m²K/W (ISO 7730)
    let icl = 0.155 * clo;

    // Metabolic rate [W/m²]
    let m = met * MET_TO_W_M2;

    // External work [W/m²]
    let w = wme * MET_TO_W_M2;

    // Internal heat production
    let mw = m - w;

    // Clothing area factor (Fanger's model, ISO 7730)
    // For low insulation (icl <= 0.078): fcl = 1.0 + 1.29 * icl
    // For higher insulation: fcl = 1.05 + 0.645 * icl
    let fcl = if icl <= 0.078 {
        1.0 + 1.29 * icl
    } else {
        1.05 + 0.645 * icl
    };

    // Heat transfer coefficient by forced convection
    // 12.1 W/(m²·K) coefficient from ISO 7730 convection formula
    let hcf = 12.1 * sqrt(vr);
    let mut hc = hcf;

    // Convert to Kelvin (using 273.0 as simplification of 273.15)
    let taa = tdb + 273.0;
    let tra = tr + 273.0;
    // Initial clothing surface temperature estimate
    // 35.5°C is approximate skin temperature, 3.5 and 0.1 are thermal resistance factors.
    // The 6.45 factor is present in the corrected ISO 7730:2025 Annex D formula but was
    // missing from the ISO 7730:2005 Annex D listing. This only shifts the starting point
    // of the iterative solve below, so converged results are unchanged.
    let tcla = taa + (35.5 - tdb) / (3.5 * (6.45 * icl + 0.1));

    // Pre-computed factors for iterative calculation
    let p1 = icl * fcl;
    let p2 = p1 * 3.96; // 3.96 is radiation coefficient (related to Stefan-Boltzmann)
    let p3 = p1 * 100.0;
    let p4 = p1 * taa;
    // 308.7 K (35.55°C) is approximate mean skin temperature
    // 0.028 is metabolic heat coefficient from Fanger's model
    let p5 = 308.7 - 0.028 * mw + p2 * pow(tra / 100.0, 4.0);

    let mut xn = tcla / 100.0;
    let mut xf = tcla / 50.0;
    // Convergence tolerance for iterative clothing temperature calculation
    let eps = 0.00015;

    let mut n = 0;
    while abs(xn - xf) > eps {
        xf = (xf + xn) / 2.0;
        // Natural convection coefficient: 2.38 W/(m²·K^1.25) and exponent 0.25
        // from natural convection heat transfer correlation
        let hcn = 2.38 * pow(abs(100.0 * xf - taa), 0.25);
        hc = fmax(hcn, hcf);
        xn = (p5 + p4 * hc - p2 * pow(xf, 4.0)) / (100.0 + p3 * hc);
        n += 1;
        // Maximum 150 iterations to prevent infinite loops
        if n > 150 {
            // Max iterations exceeded, return NaN
            return f64::NAN;
        }
    }

    let tcl = 100.0 * xn - 273.0;

    // Heat losses (all from Fanger's thermal comfort model, ISO 7730)

    // Heat loss by diffusion through skin
    // 3.05 = permeability coefficient [W/(m²·kPa)]
    // 0.001 = unit conversion factor
    // 5733 = vapor pressure constant [Pa]
    // 6.99 = metabolic rate coefficient
    let hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa);

    // Heat loss by sweating (regulatory sweating)
    // 0.42 = sweating efficiency coefficient [W/(m²) per W/(m²)]
    // Only occurs when metabolic rate exceeds 1 met (58.15 W/m²)
    let hl2 = if mw > MET_TO_W_M2 {
        0.42 * (mw - MET_TO_W_M2)
    } else {
        0.0
    };

    // Latent respiration heat loss
    // 1.7 = respiratory coefficient
    // 0.00001 = unit conversion factor
    // 5867 = vapor pressure constant for exhaled air [Pa]
    let hl3 = 1.7 * 0.00001 * m * (5867.0 - pa);

    // Dry respiration heat loss
    // 0.0014 = respiration heat coefficient [W/(m²·K) per W/m²]
    // 34°C = approximate exhaled air temperature
    let hl4 = 0.0014 * m * (34.0 - tdb);

    // Heat loss by radiation
    // 3.96 = radiation heat transfer coefficient [W/(m²·K⁴)]
    // (related to Stefan-Boltzmann constant × emissivity)
    let hl5 = 3.96 * fcl * (pow(xn, 4.0) - pow(tra / 100.0, 4.0));

    // Heat loss by convection
    let hl6 = fcl * hc * (tcl - tdb);

    // PMV calculation using thermal sensation coefficient
    // 0.303 = base thermal sensation coefficient
    // 0.036 = metabolic rate adjustment exponent (1/[W/m²])
    // 0.028 = minimum thermal sensation coefficient
    // These coefficients relate the heat balance to thermal sensation votes
    let ts = 0.303 * exp(-0.036 * m) + 0.028;
    ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
}

/// The comfort inputs to [`pmv_a`]: pythermalcomfort's five environmental parameters
/// plus `a_coefficient`, all required (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvAInputs {
    /// Dry bulb air temperature
    pub dry_bulb_temp: Temperature,
    /// Mean radiant temperature
    pub mean_radiant_temp: Temperature,
    /// Relative air speed
    pub relative_air_speed: Speed,
    /// Relative humidity
    pub relative_humidity: Humidity,
    /// Metabolic rate
    pub metabolic_rate: MetabolicRate,
    /// Clothing insulation
    pub clothing_insulation: ClothingInsulation,
    /// Adaptive coefficient (λ)
    pub a_coefficient: f64,
}

/// Optional parameters for [`pmv_a`], with pythermalcomfort's defaults.
///
/// pythermalcomfort's `pmv_a` exposes only `wme` and `limit_inputs` as optional
/// parameters — unlike [`pmv_ppd_iso`], it takes no `round_output` (its inner PMV is
/// always computed with rounding on) and no `model` (it always uses ISO 7730:2025).
/// A previous port reused the full ISO options here, which let a caller pass
/// `round_output: false` and have it silently do nothing — a parameter Python does not
/// expose at all. This narrower type forecloses that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvAOptions {
    /// External work, default 0 met
    pub wme: MetabolicRate,
    /// Limit inputs to standard compliance ranges
    pub limit_inputs: bool,
}

impl Default for PmvAOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            limit_inputs: true,
        }
    }
}

/// Calculate Adaptive Predicted Mean Vote (aPMV)
///
/// This index was developed by Yao et al. (2009) and takes into account factors such as
/// culture, climate, social, psychological, and behavioral adaptations.
///
/// # Returns
///
/// Adaptive PMV value
///
/// # Formula
///
/// aPMV = PMV / (1 + λ * PMV)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::pmv::{pmv_a, PmvAInputs};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let a_pmv = pmv_a(
///     PmvAInputs {
///         dry_bulb_temp: Temperature::from_celsius(25.0),
///         mean_radiant_temp: Temperature::from_celsius(25.0),
///         relative_air_speed: Speed::from_meters_per_second(0.1),
///         relative_humidity: Humidity::from_percent(50.0),
///         metabolic_rate: MetabolicRate::from_met(1.2),
///         clothing_insulation: ClothingInsulation::from_clo(0.5),
///         a_coefficient: 0.5,
///     },
///     Default::default(),
/// );
/// // Adaptive PMV adjusts standard PMV based on expectancy
/// ```
///
/// # References
///
/// - Yao R, Li B, Liu J (2009) Indoor Built Environ 18(5):394-411
pub fn pmv_a(inputs: PmvAInputs, options: PmvAOptions) -> f64 {
    let PmvAInputs {
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        a_coefficient,
    } = inputs;
    let PmvAOptions { wme, limit_inputs } = options;

    let pmv = pmv_ppd_iso(
        PmvPpdInputs {
            dry_bulb_temp,
            mean_radiant_temp,
            relative_air_speed,
            relative_humidity,
            metabolic_rate,
            clothing_insulation,
        },
        PmvPpdIsoOptions {
            wme,
            model: Iso7730Model::Iso77302025,
            limit_inputs,
            // pythermalcomfort's pmv_a exposes no round_output, so the inner PMV it
            // calls into is always the *rounded* ISO PMV.
            round_output: true,
        },
    )
    .pmv;
    let a_pmv = pmv / (1.0 + a_coefficient * pmv);
    crate::utilities::round_half_even(a_pmv * 100.0) / 100.0 // Round to 2 decimal places
}

/// The comfort inputs to [`pmv_e`]: pythermalcomfort's five environmental parameters
/// plus `e_coefficient`, all required (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvEInputs {
    /// Dry bulb air temperature
    pub dry_bulb_temp: Temperature,
    /// Mean radiant temperature
    pub mean_radiant_temp: Temperature,
    /// Relative air speed
    pub relative_air_speed: Speed,
    /// Relative humidity
    pub relative_humidity: Humidity,
    /// Metabolic rate
    pub metabolic_rate: MetabolicRate,
    /// Clothing insulation
    pub clothing_insulation: ClothingInsulation,
    /// Expectancy factor
    pub e_coefficient: f64,
}

/// Optional parameters for [`pmv_e`], with pythermalcomfort's defaults.
///
/// See [`PmvAOptions`]: pythermalcomfort's `pmv_e` similarly exposes only `wme` and
/// `limit_inputs`, no `round_output` and no `model`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvEOptions {
    /// External work, default 0 met
    pub wme: MetabolicRate,
    /// Limit inputs to standard compliance ranges
    pub limit_inputs: bool,
}

impl Default for PmvEOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            limit_inputs: true,
        }
    }
}

/// Calculate Adjusted PMV with Expectancy Factor (ePMV)
///
/// Developed by Fanger et al. (2002) for non-air-conditioned buildings in warm climates.
/// Accounts for occupants' low expectations in naturally ventilated spaces.
///
/// # Returns
///
/// Adjusted PMV with expectancy factor
///
/// # Formula
///
/// ePMV = PMV * e_coefficient
///
/// For warm conditions (PMV > 0), metabolic rate is adjusted:
/// met_adjusted = met * (1 + PMV * (-0.067))
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::pmv::{pmv_e, PmvEInputs};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let e_pmv = pmv_e(
///     PmvEInputs {
///         dry_bulb_temp: Temperature::from_celsius(28.0),
///         mean_radiant_temp: Temperature::from_celsius(28.0),
///         relative_air_speed: Speed::from_meters_per_second(0.2),
///         relative_humidity: Humidity::from_percent(60.0),
///         metabolic_rate: MetabolicRate::from_met(1.2),
///         clothing_insulation: ClothingInsulation::from_clo(0.5),
///         e_coefficient: 0.7,
///     },
///     Default::default(),
/// );
/// // Expectancy PMV for naturally ventilated buildings
/// ```
///
/// # References
///
/// - Fanger PO, Toftum J (2002) Energy Build 34(2):153-9
pub fn pmv_e(inputs: PmvEInputs, options: PmvEOptions) -> f64 {
    let PmvEInputs {
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        e_coefficient,
    } = inputs;
    let PmvEOptions { wme, limit_inputs } = options;

    let iso_options = PmvPpdIsoOptions {
        wme,
        model: Iso7730Model::Iso77302025,
        limit_inputs,
        // pythermalcomfort's pmv_e exposes no round_output, so the inner PMV it calls
        // into is always the *rounded* ISO PMV.
        round_output: true,
    };

    // First PMV calculation
    let pmv1 = pmv_ppd_iso(
        PmvPpdInputs {
            dry_bulb_temp,
            mean_radiant_temp,
            relative_air_speed,
            relative_humidity,
            metabolic_rate,
            clothing_insulation,
        },
        iso_options,
    )
    .pmv;

    // Adjust metabolic rate if warm (PMV > 0)
    let met_adjusted = if pmv1 > 0.0 {
        MetabolicRate::from_met(metabolic_rate.as_met() * (1.0 + pmv1 * (-0.067)))
    } else {
        metabolic_rate
    };

    // Recalculate PMV with adjusted metabolic rate
    let pmv2 = pmv_ppd_iso(
        PmvPpdInputs {
            dry_bulb_temp,
            mean_radiant_temp,
            relative_air_speed,
            relative_humidity,
            metabolic_rate: met_adjusted,
            clothing_insulation,
        },
        iso_options,
    )
    .pmv;

    let e_pmv = pmv2 * e_coefficient;
    crate::utilities::round_half_even(e_pmv * 100.0) / 100.0 // Round to 2 decimal places
}

/// The comfort inputs to [`pmv_athb`]: pythermalcomfort requires all of these (no
/// default) — `clo` is the only defaulted parameter, and lives on
/// [`PmvAthbOptions`] instead.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmvAthbInputs {
    /// Dry bulb air temperature
    pub dry_bulb_temp: Temperature,
    /// Mean radiant temperature
    pub mean_radiant_temp: Temperature,
    /// Relative air speed
    pub relative_air_speed: Speed,
    /// Relative humidity
    pub relative_humidity: Humidity,
    /// Metabolic rate
    pub metabolic_rate: MetabolicRate,
    /// Running mean outdoor temperature
    pub running_mean_outdoor_temp: Temperature,
}

/// Optional parameters for [`pmv_athb`], with pythermalcomfort's default.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PmvAthbOptions {
    /// Clothing insulation. `None` (pythermalcomfort's `clo=False`) derives it from the
    /// running mean outdoor temperature via behavioral adaptation instead of taking a
    /// caller-supplied value.
    pub clothing_insulation: Option<ClothingInsulation>,
}

/// Calculate PMV using Adaptive Thermal Heat Balance (ATHB) framework
///
/// Developed by Schweiker et al. (2022). Accounts for physiological, behavioral,
/// and psychological adaptation within heat balance models.
///
/// # Returns
///
/// ATHB-adjusted PMV value
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::pmv::{pmv_athb, PmvAthbInputs, PmvAthbOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let athb_pmv = pmv_athb(
///     PmvAthbInputs {
///         dry_bulb_temp: Temperature::from_celsius(25.0),
///         mean_radiant_temp: Temperature::from_celsius(25.0),
///         relative_air_speed: Speed::from_meters_per_second(0.1),
///         relative_humidity: Humidity::from_percent(50.0),
///         metabolic_rate: MetabolicRate::from_met(1.2),
///         running_mean_outdoor_temp: Temperature::from_celsius(20.0),
///     },
///     PmvAthbOptions {
///         clothing_insulation: Some(ClothingInsulation::from_clo(0.6)),
///     },
/// );
/// // ATHB PMV accounts for physiological and behavioral adaptation
/// ```
///
/// # References
///
/// - Schweiker M et al. (2022) Build Environ 216:109017
pub fn pmv_athb(inputs: PmvAthbInputs, options: PmvAthbOptions) -> f64 {
    let PmvAthbInputs {
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        running_mean_outdoor_temp,
    } = inputs;
    let PmvAthbOptions {
        clothing_insulation,
    } = options;

    let running_mean_celsius = running_mean_outdoor_temp.as_celsius();

    // Adapt metabolic rate for psychological adaptation
    let met_adapted = metabolic_rate.as_met() - (0.234 * running_mean_celsius) / 58.2;

    // Calculate or use provided clothing insulation
    let clo_adapted = if let Some(c) = clothing_insulation {
        c.as_clo()
    } else {
        // Behavioral adaptation: calculate clothing from conditions
        let exponent = -0.17168 - 0.000485 * running_mean_celsius + 0.08176 * met_adapted
            - 0.00527 * running_mean_celsius * met_adapted;
        libm::pow(10.0, exponent)
    };

    // Calculate base PMV with adapted parameters
    let pmv_result = pmv_ppd_iso(
        PmvPpdInputs {
            dry_bulb_temp,
            mean_radiant_temp,
            relative_air_speed,
            relative_humidity,
            metabolic_rate: MetabolicRate::from_met(met_adapted),
            clothing_insulation: ClothingInsulation::from_clo(clo_adapted),
        },
        PmvPpdIsoOptions {
            limit_inputs: false, // ATHB may use values outside standard limits
            // Python calls the raw _pmv_ppd_optimized kernel here, so the inner PMV must
            // not be rounded before being divided by ts.
            round_output: false,
            ..Default::default()
        },
    )
    .pmv;

    // Calculate thermal sensation coefficient
    // 58.15 here, not the 58.2 used above for met_adapted: pythermalcomfort uses the
    // met_to_w_m2 constant in this expression and a literal 58.2 in the other.
    let ts = 0.303 * libm::exp(-0.036 * met_adapted * MET_TO_W_M2) + 0.028;
    let l_adapted = pmv_result / ts;

    // Calculate ATHB PMV
    let athb_pmv =
        1.484 + 0.0276 * l_adapted - 0.9602 * met_adapted - 0.0342 * running_mean_celsius
            + 0.0002264 * l_adapted * running_mean_celsius
            + 0.018696 * met_adapted * running_mean_celsius
            - 0.0002909 * l_adapted * met_adapted * running_mean_celsius;

    crate::utilities::round_half_even(athb_pmv * 1000.0) / 1000.0 // Round to 3 decimal places
}
#[cfg(test)]
mod tests {
    use super::*;

    fn pmv_inputs(tdb: f64, tr: f64, vr: f64, rh: f64, met: f64, clo: f64) -> PmvPpdInputs {
        PmvPpdInputs {
            dry_bulb_temp: Temperature::from_celsius(tdb),
            mean_radiant_temp: Temperature::from_celsius(tr),
            relative_air_speed: Speed::from_meters_per_second(vr),
            relative_humidity: Humidity::from_percent(rh),
            metabolic_rate: MetabolicRate::from_met(met),
            clothing_insulation: ClothingInsulation::from_clo(clo),
        }
    }

    #[test]
    fn test_thermal_sensation_mapping() {
        for (pmv, expected) in [
            (-3.0, ThermalSensation::Cold),
            (-2.0, ThermalSensation::Cool),
            (-1.0, ThermalSensation::SlightlyCool),
            (0.0, ThermalSensation::Neutral),
            (1.0, ThermalSensation::SlightlyWarm),
            (2.0, ThermalSensation::Warm),
            (3.0, ThermalSensation::Hot),
        ] {
            assert_eq!(ThermalSensation::from_pmv_opt(pmv), Some(expected));
        }

        // A PMV that was never computed has no band
        assert_eq!(ThermalSensation::from_pmv_opt(f64::NAN), None);
    }

    #[test]
    fn test_pmv_ppd_iso_basic() {
        // Example from ISO 7730
        let result = pmv_ppd_iso(
            pmv_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );

        // Should be approximately neutral comfort
        assert!(result.pmv.abs() < 0.5);
        assert!(result.ppd < 10.0);
        // ISO never populates the ASHRAE compliance field.
        assert_eq!(result.compliance, None);
    }

    #[test]
    fn test_pmv_ppd_ashrae_cooling_effect_triggers() {
        // With vr > 0.1, ASHRAE Appendix H3 cooling-effect correction kicks in,
        // lowering effective tdb/tr and pulling PMV closer to neutral.
        let base = pmv_ppd_ashrae(
            pmv_inputs(28.0, 28.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );
        let breezy = pmv_ppd_ashrae(
            pmv_inputs(28.0, 28.0, 0.8, 50.0, 1.2, 0.5),
            Default::default(),
        );
        // Warm conditions: elevated airspeed must reduce PMV.
        assert!(
            breezy.pmv < base.pmv,
            "expected cooling effect to lower PMV: base={}, breezy={}",
            base.pmv,
            breezy.pmv,
        );
    }

    #[test]
    fn test_pmv_ppd_ashrae_compliance() {
        // Compliant case: PMV near 0 should be within -0.5..0.5.
        let result = pmv_ppd_ashrae(
            pmv_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );
        assert_eq!(result.compliance, Some(true));

        // Non-compliant case: hot conditions push PMV beyond +0.5.
        let result = pmv_ppd_ashrae(
            pmv_inputs(30.0, 30.0, 0.1, 60.0, 1.4, 0.3),
            Default::default(),
        );
        assert_eq!(result.compliance, Some(false));

        // Out-of-applicability case: limit_inputs=true and tdb < 10 → NaN/None.
        let result = pmv_ppd_ashrae(
            pmv_inputs(5.0, 5.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );
        assert!(result.pmv.is_nan());
        assert_eq!(result.compliance, None);
    }

    #[test]
    fn test_pmv_ppd_iso_limits() {
        // Test with values outside limits
        let result = pmv_ppd_iso(
            pmv_inputs(5.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );
        assert!(result.pmv.is_nan());
        assert!(result.ppd.is_nan());

        // Test with limits disabled
        let options = PmvPpdIsoOptions {
            limit_inputs: false,
            ..Default::default()
        };
        let result = pmv_ppd_iso(pmv_inputs(5.0, 25.0, 0.1, 50.0, 1.2, 0.5), options);
        assert!(!result.pmv.is_nan());
    }

    /// `airspeed_control` gates ASHRAE 55's §7.2.1.2 cross-variable rules. This case
    /// (tdb=tr=24°C so `to`=24, inside the 23-25.5°C comfort band; clo=0.3 < 0.7;
    /// met=1.0 < 1.3) has `v_limit = 50.49 - 4.4047*24 + 0.096425*24^2 ≈ 0.318 m/s`. At
    /// vr=0.5 m/s, that limit is exceeded, but 0.5 is still within the flat
    /// Table 7.3.4 range (0-2 m/s) that `airspeed_control=true` (upstream's default)
    /// alone checks. Only when the occupant is assumed *not* to control airspeed does
    /// §7.2.1.2's tighter, `to`-dependent limit apply — that path was previously
    /// unreachable from Rust because `airspeed_control` was not exposed at all.
    #[test]
    fn test_pmv_ppd_ashrae_airspeed_control_false_diverges() {
        let inputs = pmv_inputs(24.0, 24.0, 0.5, 50.0, 1.0, 0.3);

        let controlled = pmv_ppd_ashrae(inputs, Default::default());
        assert!(
            !controlled.pmv.is_nan(),
            "airspeed_control=true (default) should accept vr=0.5 m/s here"
        );

        let uncontrolled = pmv_ppd_ashrae(
            inputs,
            PmvPpdAshraeOptions {
                airspeed_control: false,
                ..Default::default()
            },
        );
        assert!(
            uncontrolled.pmv.is_nan(),
            "airspeed_control=false should reject vr=0.5 m/s inside the comfort band \
             with clo<0.7 and met<1.3, but got pmv={}",
            uncontrolled.pmv
        );
        assert_eq!(uncontrolled.compliance, None);
    }
}
