//! Reading and building quantities in SI or IP (inch-pound) units.
//!
//! Every quantity type the crate uses is unit-safe by construction: a [`Temperature`] is one
//! physical temperature, whatever scale it was built from. [`UnitSystem`] adds one pair of
//! accessors per type for the two unit systems pythermalcomfort's `units` option names, so a
//! caller working entirely in SI or entirely in IP never picks a unit per quantity:
//!
//! ```
//! use thermalcomfort::{Speed, Temperature, UnitSystem};
//!
//! let t = Temperature::from_ip(77.0); // °F
//! assert!((t.as_si() - 25.0).abs() < 1e-9); // °C
//!
//! let v = Speed::from_si(1.0); // m/s
//! assert!((v.as_ip() - 3.281).abs() < 1e-12); // ft/s
//! println!("{:.1} {} = {:.2} {}", v.as_si(), Speed::SI_UNIT, v.as_ip(), Speed::IP_UNIT);
//! ```
//!
//! **IP means inch-pound**, the US customary system ASHRAE publishes in: °F, ft, ft², ft/s,
//! lb, psi and Btu. It is not the British imperial system, though the two agree on these
//! units.
//!
//! The SI unit is the one pythermalcomfort uses for the quantity, so `as_si()` gives
//! upstream's number. Where a quantity has several SI scales, `as_si` uses the one named
//! here and the type's named accessors give the others.
//!
//! | Type | SI unit (`as_si`) | IP (inch-pound) unit (`as_ip`) |
//! |---|---|---|
//! | [`Temperature`] | °C (`as_celsius`) | °F (`as_fahrenheit`) |
//! | [`TemperatureDelta`] | °C (= K) | °F (× 1.8, no offset) |
//! | [`Speed`] | m/s (`as_meters_per_second`) | ft/s (× 3.281, pythermalcomfort's factor) |
//! | [`Length`] | m (`as_meters`) | ft (`as_feet`) |
//! | [`Area`] | m² (`as_square_meters`) | ft² (`as_square_feet`) |
//! | [`Mass`] | kg (`as_kilograms`) | lb (`as_pounds`) |
//! | [`Pressure`] | Pa (`as_pascals`) | psi (`as_psi`) |
//! | [`Power`] | W (`as_watts`) | Btu/h (`as_btu_per_minute` × 60) |
//! | [`HeatFluxDensity`] | W/m² | Btu/(h·ft²) |
//! | [`MetabolicRate`] | W/m² (met via `as_met`) | Btu/(h·ft²) (ASHRAE's 1 met = 58.15 W/m² = 18.4 Btu/(h·ft²)) |
//! | [`ClothingInsulation`] | m²·K/W (clo, tog via `as_clo`, `as_tog`) | ft²·h·°F/Btu |
//! | [`AirPermeability`] | L/(m²·s) | ft³/(min·ft²) |
//! | [`MassFluxDensity`] | g/(h·m²) | lb/(h·ft²) |
//! | [`VolumeFluxDensity`] | L/(h·m²) | gal/(h·ft²) (US gallon, 3.785 411 784 L) |
//! | [`VolumeFlow`] | L/h | gal/h (US gallon) |
//! | [`MassFlow`] | g/s | lb/h |
//! | [`SpecificEnthalpy`] | J/kg | Btu/lb |
//! | [`HumidityRatio`] | kg/kg (g/kg via `as_grams_per_kilogram`) | lb/lb (the same number; gr/lb via `as_grains_per_pound`) |
//! | [`HeatPerArea`] | J/m² (W·min/m² via `as_watt_minutes_per_square_meter`) | Btu/ft² |
//! | [`ThermalResistance`] | m²·K/W | ft²·h·°F/Btu |
//! | [`EvaporativeResistance`] | m²·kPa/W | ft²·h·psi/Btu |
//! | [`PowerRate`] | W/s | Btu/(h·s) |
//! | [`SweatingCoefficient`] | g/(m²·h·K) | lb/(ft²·h·°F) |
//! | [`VasodilationCoefficient`] | L/(m²·h·K) | gal/(ft²·h·°F) (US gallon) |
//! | [`VasoconstrictionCoefficient`] | 1/K | 1/°F |
//!
//! # Repeatability: conversion factors and round-off
//!
//! A result converted between the systems is reproducible only if the conversion factor
//! is known. These are the factors this crate uses. Most are exact; four base factors are
//! not, every conversion built on one of them inherits its error, and the table states how
//! far each is from the exact value, so a result can be reproduced or cross-checked
//! elsewhere to the last digit. "Exact" means the factor is the exact one, rounded once to
//! `f64` (within 4e-16 relative).
//!
//! | Conversion | Factor used | Exact? | Difference from exact |
//! |---|---|---|---|
//! | °C → °F | `× 9/5 + 32` (via kelvin, see below) | yes | — |
//! | °C difference → °F difference | `× 1.8` | yes | — |
//! | m → ft | `/ 0.3048` | yes (definition) | — |
//! | m² → ft² | `/ 0.3048²` | yes | — |
//! | kg → lb | `/ 0.45359237` | yes (definition) | — |
//! | **m/s → ft/s** | **`× 3.281`** | **no** | **+4.9e-5 relative (exact: 3.280840)** |
//! | **Pa → psi** | **`/ 6894.76`** | **no** | **−3.9e-7 relative (exact: 6894.757293 Pa)** |
//! | **W → Btu/h** | **`× 60 / 17.58426667`** | **no** | **−1.4e-7 relative (IT Btu: 1055.05585 J)** |
//! | W/m² → Btu/(h·ft²) | the W and m² factors above | as W → Btu/h | −1.4e-7 relative |
//! | **met → Btu/(h·ft²)** | **`× 18.4`** (and `× 58.15` to W/m²) | **no** | **−0.18 % (58.15 W/m² is 18.433 Btu/(h·ft²))** |
//! | m²·K/W → ft²·h·°F/Btu | the ft², °F and Btu factors above | as W → Btu/h | +1.4e-7 relative |
//! | L/(m²·s) → ft³/(min·ft²) | `× 0.001 × 60 / 0.3048` | yes | — |
//! | g/(h·m²) → lb/(h·ft²) | `× 0.001 / 0.45359237 × 0.3048²` | yes | — |
//! | L/(h·m²) → gal/(h·ft²) | `/ 3.785411784 × 0.3048²` | yes (US gallon, definition) | — |
//! | L/h → gal/h | `/ 3.785411784` | yes | — |
//! | g/s → lb/h | `× 0.001 / 0.45359237 × 3600` | yes | — |
//! | J/kg → Btu/lb | the Btu factor above, `/ 3600 × 0.45359237` | as W → Btu/h | −1.4e-7 relative (exact: 1 Btu/lb = 2326 J/kg) |
//! | kg/kg → lb/lb | `× 1` | yes | — |
//! | kg/kg → g/kg | `× 1000` | yes | — |
//! | kg/kg → gr/lb | `× 7000` | yes (definition) | — |
//! | J/m² → Btu/ft² | the Btu factor above, `/ 3600 × 0.3048²` | as W → Btu/h | −1.4e-7 relative |
//! | J/m² → W·min/m² | `/ 60` | yes | — |
//! | m²·K/W → ft²·h·°F/Btu ([`ThermalResistance`]) | as [`ClothingInsulation`] | as W → Btu/h | +1.4e-7 relative |
//! | **m²·kPa/W → ft²·h·psi/Btu** | the ft², psi and Btu factors above | **no** | **−2.5e-7 relative (psi −3.9e-7 and Btu +1.4e-7 combined)** |
//! | W/s → Btu/(h·s) | the Btu factor above | as W → Btu/h | −1.4e-7 relative |
//! | g/(m²·h·K) → lb/(ft²·h·°F) | `× 0.001 / 0.45359237 × 0.3048² / 1.8` | yes | — |
//! | L/(m²·h·K) → gal/(ft²·h·°F) | `/ 3.785411784 × 0.3048² / 1.8` | yes | — |
//! | 1/K → 1/°F | `/ 1.8` | yes | — |
//!
//! Why the four inexact factors are used anyway:
//!
//! - **Speed, 3.281.** pythermalcomfort converts every IP air speed with
//!   `units_converter`'s `v / 3.281`, so `Speed::from_ip(x)` is the speed upstream means by
//!   an IP input of `x`. With the exact factor, an IP user's results would differ from
//!   pythermalcomfort's by up to 5e-5 relative in air speed. To convert a measured ft/s
//!   exactly, use `Speed::from_meters_per_second(x * 0.3048)` instead.
//! - **Pressure, 6894.76 Pa/psi**, and **power, 17.58426667 W per Btu/min**: the
//!   `measurements` crate's own constants, so `as_ip` always equals `Pressure::as_psi` and
//!   `Power::as_btu_per_minute() × 60`. Both are well below any model's precision.
//! - **Metabolic rate, 18.4 Btu/(h·ft²) per met**: ASHRAE 55's tabulated value, which
//!   [`MetabolicRate::as_btu_per_h_ft2`] has always used. The same flux built as a
//!   [`HeatFluxDensity`] converts exactly, so 1 met reads 18.4 as a metabolic rate and
//!   18.433 as a heat flux.
//!
//! Three sources of round-off besides the factors:
//!
//! - **Temperatures are stored in kelvin** (by the `measurements` crate), so a Celsius or
//!   Fahrenheit value read back can differ from the one stored by about 3e-14 °C (5e-14 °F),
//!   the last bit of the kelvin value. Integers and half-integers are exact. Model code that
//!   must match upstream bit-for-bit converts plain `f64` values with upstream's own
//!   `(value * 9 / 5) + 32` instead.
//! - **A model's `units` option selects the unit its result is rounded in**, as
//!   pythermalcomfort's does: the cooling effect to 0.01, the adaptive comfort
//!   temperatures, UTCI and the running mean outdoor temperature to 0.1, each in °C for
//!   `SI` or °F for `IP`. Read the result in the same system you passed to get upstream's
//!   rounded number, to within the storage round-off above (a temperature difference
//!   read in °F is stored in °C, so it carries the same few-ULP round-off). Reading it in the other system is the same physical value,
//!   unrounded: an `IP` cooling effect of 6.62 °F reads `as_si()` as 3.6778 °C, not 3.68.
//! - **Rounding rules follow upstream's, which use two.** Values rounded by numpy, which is
//!   most of them, use round-half-to-even on `value × 10ⁿ`. The running mean outdoor
//!   temperature is rounded by CPython's builtin `round`, which rounds the exact decimal
//!   value of the float: `-22.285` (really `-22.28500000000000085…`) rounds to `-22.29`
//!   there and to `-22.28` under numpy's rule.
//!
//! Types with no separate IP unit have no implementation: [`Humidity`](crate::Humidity)
//! (%), [`Angle`](crate::Angle) (degrees), durations, ages, and the dimensionless
//! fractions and ratios.

use crate::utilities::Units;
use crate::{
    AirPermeability, Area, ClothingInsulation, EvaporativeResistance, HeatFluxDensity, HeatPerArea,
    HumidityRatio, Length, Mass, MassFlow, MassFluxDensity, MetabolicRate, Power, PowerRate,
    Pressure, SpecificEnthalpy, Speed, SweatingCoefficient, Temperature, TemperatureDelta,
    ThermalResistance, VasoconstrictionCoefficient, VasodilationCoefficient, VolumeFlow,
    VolumeFluxDensity,
};
use measurements::pressure::PASCAL_PSI_FACTOR;

/// Feet per metre, exact (1 ft = 0.3048 m).
const FT_PER_M: f64 = 1.0 / 0.3048;
/// Square feet per square metre, exact.
const FT2_PER_M2: f64 = FT_PER_M * FT_PER_M;
/// Pounds per kilogram, exact (1 lb = 0.453 592 37 kg).
const LB_PER_KG: f64 = 1.0 / 0.453_592_37;
/// US gallons per litre, exact (1 gal = 3.785 411 784 L).
const GAL_PER_L: f64 = 1.0 / 3.785_411_784;
/// Btu/h per watt, from the same Btu as [`Power`]'s `as_btu_per_minute`.
const BTU_H_PER_W: f64 = 60.0 / 17.584_266_666_666_67;
/// Btu per joule, the Btu of [`BTU_H_PER_W`] (1 W·h = 3600 J).
const BTU_PER_J: f64 = BTU_H_PER_W / 3600.0;
/// psi per kPa, from the same Pa-per-psi constant as [`Pressure`]'s `as_psi`.
const PSI_PER_KPA: f64 = 1000.0 / PASCAL_PSI_FACTOR;
/// °F of difference per K of difference, exact.
const F_PER_K: f64 = 1.8;
/// ft²·h·°F/Btu per m²·K/W: ft² per m², °F per K, and the inverse of Btu/h per W.
const THERMAL_RESISTANCE_IP_PER_SI: f64 = FT2_PER_M2 * F_PER_K / BTU_H_PER_W;
/// pythermalcomfort's ft/s per m/s (`units_converter`'s `v * 3.281`).
const FT_S_PER_M_S: f64 = 3.281;

/// A quantity that can be read and built in SI or IP (inch-pound) units.
///
/// See the [module documentation](crate::unit_system) for the unit of each type.
pub trait UnitSystem: Sized + Copy {
    /// Symbol of the SI unit, for display
    const SI_UNIT: &'static str;
    /// Symbol of the IP (inch-pound) unit, for display
    const IP_UNIT: &'static str;

    /// Build from a value in the SI unit
    fn from_si(value: f64) -> Self;
    /// Build from a value in the IP (inch-pound) unit
    fn from_ip(value: f64) -> Self;
    /// The value in the SI unit
    fn as_si(self) -> f64;
    /// The value in the IP (inch-pound) unit
    fn as_ip(self) -> f64;

    /// Build from a value in the unit of `units`
    fn from_units(value: f64, units: Units) -> Self {
        match units {
            Units::SI => Self::from_si(value),
            Units::IP => Self::from_ip(value),
        }
    }

    /// The value in the unit of `units`
    fn in_units(self, units: Units) -> f64 {
        match units {
            Units::SI => self.as_si(),
            Units::IP => self.as_ip(),
        }
    }

    /// Symbol of the unit of `units`, for display
    fn unit(units: Units) -> &'static str {
        match units {
            Units::SI => Self::SI_UNIT,
            Units::IP => Self::IP_UNIT,
        }
    }
}

/// Implements [`UnitSystem`] from a type's own SI constructor and accessor, with the IP
/// value a fixed multiple of the SI one.
macro_rules! linear_unit_system {
    ($ty:ty, $si:literal, $ip:literal, $from_si:ident, $as_si:ident, $ip_per_si:expr) => {
        impl UnitSystem for $ty {
            const SI_UNIT: &'static str = $si;
            const IP_UNIT: &'static str = $ip;
            fn from_si(value: f64) -> Self {
                <$ty>::$from_si(value)
            }
            fn from_ip(value: f64) -> Self {
                <$ty>::$from_si(value / ($ip_per_si))
            }
            fn as_si(self) -> f64 {
                self.$as_si()
            }
            fn as_ip(self) -> f64 {
                self.$as_si() * ($ip_per_si)
            }
        }
    };
}

impl UnitSystem for Temperature {
    const SI_UNIT: &'static str = "°C";
    const IP_UNIT: &'static str = "°F";
    fn from_si(value: f64) -> Self {
        Self::from_celsius(value)
    }
    fn from_ip(value: f64) -> Self {
        Self::from_fahrenheit(value)
    }
    fn as_si(self) -> f64 {
        self.as_celsius()
    }
    fn as_ip(self) -> f64 {
        self.as_fahrenheit()
    }
}

impl UnitSystem for TemperatureDelta {
    const SI_UNIT: &'static str = "°C";
    const IP_UNIT: &'static str = "°F";
    fn from_si(value: f64) -> Self {
        Self::from_celsius(value)
    }
    fn from_ip(value: f64) -> Self {
        Self::from_fahrenheit(value)
    }
    fn as_si(self) -> f64 {
        self.as_celsius()
    }
    fn as_ip(self) -> f64 {
        self.as_fahrenheit()
    }
}

impl UnitSystem for Pressure {
    const SI_UNIT: &'static str = "Pa";
    const IP_UNIT: &'static str = "psi";
    fn from_si(value: f64) -> Self {
        Self::from_pascals(value)
    }
    // The type's own psi conversion, so `as_ip` and `as_psi` always agree
    fn from_ip(value: f64) -> Self {
        Self::from_psi(value)
    }
    fn as_si(self) -> f64 {
        self.as_pascals()
    }
    fn as_ip(self) -> f64 {
        self.as_psi()
    }
}

impl UnitSystem for MetabolicRate {
    const SI_UNIT: &'static str = "W/m²";
    const IP_UNIT: &'static str = "Btu/(h·ft²)";
    fn from_si(value: f64) -> Self {
        Self::from_w_per_m2(value)
    }
    fn from_ip(value: f64) -> Self {
        Self::from_btu_per_h_ft2(value)
    }
    fn as_si(self) -> f64 {
        self.as_w_per_m2()
    }
    fn as_ip(self) -> f64 {
        self.as_btu_per_h_ft2()
    }
}

linear_unit_system!(
    Speed,
    "m/s",
    "ft/s",
    from_meters_per_second,
    as_meters_per_second,
    FT_S_PER_M_S
);
linear_unit_system!(Length, "m", "ft", from_meters, as_meters, FT_PER_M);
linear_unit_system!(
    Area,
    "m²",
    "ft²",
    from_square_meters,
    as_square_meters,
    FT2_PER_M2
);
linear_unit_system!(Mass, "kg", "lb", from_kilograms, as_kilograms, LB_PER_KG);
linear_unit_system!(Power, "W", "Btu/h", from_watts, as_watts, BTU_H_PER_W);
linear_unit_system!(
    HeatFluxDensity,
    "W/m²",
    "Btu/(h·ft²)",
    from_watts_per_square_meter,
    as_watts_per_square_meter,
    BTU_H_PER_W / FT2_PER_M2
);
linear_unit_system!(
    ClothingInsulation,
    "m²·K/W",
    "ft²·h·°F/Btu",
    from_m2_k_per_w,
    as_m2_k_per_w,
    THERMAL_RESISTANCE_IP_PER_SI
);
linear_unit_system!(
    AirPermeability,
    "L/(m²·s)",
    "ft³/(min·ft²)",
    from_l_per_m2_s,
    as_l_per_m2_s,
    // L/(m²·s) is mm/s; ft³/(min·ft²) is ft/min
    0.001 * FT_PER_M * 60.0
);
linear_unit_system!(
    MassFluxDensity,
    "g/(h·m²)",
    "lb/(h·ft²)",
    from_grams_per_hour_per_square_meter,
    as_grams_per_hour_per_square_meter,
    LB_PER_KG / 1000.0 / FT2_PER_M2
);
linear_unit_system!(
    VolumeFluxDensity,
    "L/(h·m²)",
    "gal/(h·ft²)",
    from_liters_per_hour_per_square_meter,
    as_liters_per_hour_per_square_meter,
    GAL_PER_L / FT2_PER_M2
);
linear_unit_system!(
    VolumeFlow,
    "L/h",
    "gal/h",
    from_liters_per_hour,
    as_liters_per_hour,
    GAL_PER_L
);
linear_unit_system!(
    MassFlow,
    "g/s",
    "lb/h",
    from_grams_per_second,
    as_grams_per_second,
    // kg per g, lb per kg, s per h
    0.001 * LB_PER_KG * 3600.0
);
linear_unit_system!(
    SpecificEnthalpy,
    "J/kg",
    "Btu/lb",
    from_joules_per_kilogram,
    as_joules_per_kilogram,
    BTU_PER_J / LB_PER_KG
);
linear_unit_system!(
    HumidityRatio,
    "kg/kg",
    "lb/lb",
    from_kilograms_per_kilogram,
    as_kilograms_per_kilogram,
    // a mass ratio is the same number in any mass unit
    1.0
);
linear_unit_system!(
    HeatPerArea,
    "J/m²",
    "Btu/ft²",
    from_joules_per_square_meter,
    as_joules_per_square_meter,
    BTU_PER_J / FT2_PER_M2
);
linear_unit_system!(
    ThermalResistance,
    "m²·K/W",
    "ft²·h·°F/Btu",
    from_square_meter_kelvins_per_watt,
    as_square_meter_kelvins_per_watt,
    THERMAL_RESISTANCE_IP_PER_SI
);
linear_unit_system!(
    EvaporativeResistance,
    "m²·kPa/W",
    "ft²·h·psi/Btu",
    from_square_meter_kilopascals_per_watt,
    as_square_meter_kilopascals_per_watt,
    // ft² per m², psi per kPa, and the inverse of Btu/h per W
    FT2_PER_M2 * PSI_PER_KPA / BTU_H_PER_W
);
linear_unit_system!(
    PowerRate,
    "W/s",
    "Btu/(h·s)",
    from_watts_per_second,
    as_watts_per_second,
    BTU_H_PER_W
);
linear_unit_system!(
    SweatingCoefficient,
    "g/(m²·h·K)",
    "lb/(ft²·h·°F)",
    from_grams_per_square_meter_per_hour_per_kelvin,
    as_grams_per_square_meter_per_hour_per_kelvin,
    // per kelvin becomes per °F by dividing by 1.8
    0.001 * LB_PER_KG / FT2_PER_M2 / F_PER_K
);
linear_unit_system!(
    VasodilationCoefficient,
    "L/(m²·h·K)",
    "gal/(ft²·h·°F)",
    from_liters_per_square_meter_per_hour_per_kelvin,
    as_liters_per_square_meter_per_hour_per_kelvin,
    GAL_PER_L / FT2_PER_M2 / F_PER_K
);
linear_unit_system!(
    VasoconstrictionCoefficient,
    "1/K",
    "1/°F",
    from_per_kelvin,
    as_per_kelvin,
    1.0 / F_PER_K
);

/// A Celsius value in the unit of `units`, by pythermalcomfort's `units_converter`
/// formula `(value * 9 / 5) + 32`.
///
/// For model code that rounds in the output unit and must reproduce upstream's arithmetic
/// exactly: going through [`Temperature`]'s kelvin storage instead shifts the last bit,
/// which moves a value sitting on a rounding boundary.
pub(crate) fn celsius_in_units(value_celsius: f64, units: Units) -> f64 {
    match units {
        Units::SI => value_celsius,
        Units::IP => value_celsius * 9.0 / 5.0 + 32.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T: UnitSystem>(value: f64) {
        assert!((T::from_ip(value).as_ip() - value).abs() < 1e-9 * value.abs().max(1.0));
        assert!((T::from_si(value).as_si() - value).abs() < 1e-9 * value.abs().max(1.0));
    }

    #[test]
    fn every_type_round_trips_in_both_systems() {
        for x in [0.0, 1.0, 37.5, 101_325.0] {
            round_trip::<Temperature>(x);
            round_trip::<TemperatureDelta>(x);
            round_trip::<Speed>(x);
            round_trip::<Length>(x);
            round_trip::<Area>(x);
            round_trip::<Mass>(x);
            round_trip::<Pressure>(x);
            round_trip::<Power>(x);
            round_trip::<HeatFluxDensity>(x);
            round_trip::<MetabolicRate>(x);
            round_trip::<ClothingInsulation>(x);
            round_trip::<AirPermeability>(x);
            round_trip::<MassFluxDensity>(x);
            round_trip::<VolumeFluxDensity>(x);
            round_trip::<VolumeFlow>(x);
            round_trip::<MassFlow>(x);
            round_trip::<SpecificEnthalpy>(x);
            round_trip::<HumidityRatio>(x);
            round_trip::<HeatPerArea>(x);
            round_trip::<ThermalResistance>(x);
            round_trip::<EvaporativeResistance>(x);
            round_trip::<PowerRate>(x);
            round_trip::<SweatingCoefficient>(x);
            round_trip::<VasodilationCoefficient>(x);
            round_trip::<VasoconstrictionCoefficient>(x);
        }
    }

    #[test]
    fn known_equivalences() {
        let close = |a: f64, b: f64, tol: f64| assert!((a - b).abs() < tol, "{a} vs {b}");
        close(Temperature::from_si(100.0).as_ip(), 212.0, 1e-9);
        close(TemperatureDelta::from_si(10.0).as_ip(), 18.0, 1e-12);
        close(Speed::from_si(2.0).as_ip(), 6.562, 1e-12);
        close(Length::from_ip(1.0).as_si(), 0.3048, 1e-15);
        close(Area::from_ip(1.0).as_si(), 0.092_903_04, 1e-15);
        close(Mass::from_ip(1.0).as_si(), 0.453_592_37, 1e-15);
        close(Pressure::from_si(101_325.0).as_ip(), 14.695_95, 1e-4);
        close(Power::from_si(1.0).as_ip(), 3.412_14, 1e-5);
        close(HeatFluxDensity::from_ip(1.0).as_si(), 3.154_59, 1e-5);
        close(MetabolicRate::from_met(1.0).as_ip(), 18.4, 1e-12);
        close(ClothingInsulation::from_clo(1.0).as_ip(), 0.880_1, 1e-4);
        close(AirPermeability::from_ip(1.0).as_si(), 5.08, 1e-12);
        close(MassFluxDensity::from_ip(1.0).as_si(), 4_882.427_6, 1e-4);
        close(VolumeFluxDensity::from_ip(1.0).as_si(), 40.745_833, 1e-6);
        close(VolumeFlow::from_ip(1.0).as_si(), 3.785_411_784, 1e-12);
        close(MassFlow::from_ip(1.0).as_si(), 0.125_997_88, 1e-8);
        close(SpecificEnthalpy::from_ip(1.0).as_si(), 2326.0, 1e-3);
        close(HumidityRatio::from_ip(0.01).as_si(), 0.01, 1e-15);
        close(HeatPerArea::from_ip(1.0).as_si(), 11_356.53, 1e-2);
        close(ThermalResistance::from_si(1.0).as_ip(), 5.678_263, 1e-5);
        close(EvaporativeResistance::from_ip(1.0).as_si(), 2.185_627, 1e-5);
        close(PowerRate::from_si(1.0).as_ip(), 3.412_14, 1e-5);
        close(SweatingCoefficient::from_ip(1.0).as_si(), 8_788.37, 1e-2);
        close(VasodilationCoefficient::from_ip(1.0).as_si(), 73.3425, 1e-9);
        close(
            VasoconstrictionCoefficient::from_si(1.8).as_ip(),
            1.0,
            1e-15,
        );
        // the named accessors agree with as_si where a type has several SI scales
        let w = HumidityRatio::from_grams_per_kilogram(12.0);
        close(w.as_si(), 0.012, 1e-15);
        close(w.as_grams_per_kilogram(), 12.0, 1e-12);
        let q = HeatPerArea::from_watt_minutes_per_square_meter(3.0);
        close(q.as_si(), 180.0, 1e-12);
        // W·min/m² is held directly, so PHS's accumulator passes through unchanged; at
        // this value `× 60 / 60` is not the identity
        let load = 1348.473;
        assert_ne!(load * 60.0 / 60.0, load);
        assert_eq!(
            HeatPerArea::from_watt_minutes_per_square_meter(load)
                .as_watt_minutes_per_square_meter(),
            load
        );
        // ThermalResistance and ClothingInsulation share the IP unit and factor
        let clo = ClothingInsulation::from_clo(1.0);
        close(
            ThermalResistance::from_si(clo.as_si()).as_ip(),
            clo.as_ip(),
            1e-15,
        );
        // the measurements crate's own accessors agree where they exist
        let p = Pressure::from_psi(14.7);
        close(p.as_ip(), p.as_psi(), 1e-9);
        let a = Area::from_square_meters(1.8);
        close(a.as_ip(), a.as_square_feet(), 1e-12);
        let l = Length::from_meters(1.71);
        close(l.as_ip(), l.as_feet(), 1e-12);
        let m = Mass::from_kilograms(70.0);
        close(m.as_ip(), m.as_pounds(), 1e-9);
    }

    /// Pins the module documentation's repeatability table: each inexact factor, and its
    /// stated difference from the exact conversion.
    #[test]
    fn documented_factors_and_their_error() {
        let rel = |got: f64, exact: f64| got / exact - 1.0;
        let close = |a: f64, b: f64, tol: f64| assert!((a - b).abs() < tol, "{a} vs {b}");
        // m/s -> ft/s: 3.281, +4.9e-5 relative
        close(Speed::from_si(1.0).as_ip(), 3.281, 1e-15);
        close(
            rel(Speed::from_si(1.0).as_ip(), 1.0 / 0.3048),
            4.88e-5,
            1e-7,
        );
        // Pa -> psi: 6894.76 Pa per psi, -3.9e-7 relative
        close(Pressure::from_ip(1.0).as_si(), 6894.76, 1e-9);
        close(
            rel(Pressure::from_si(1.0).as_ip(), 1.0 / 6_894.757_293_168),
            -3.9e-7,
            1e-8,
        );
        // W -> Btu/h: -1.4e-7 relative to the IT Btu (1055.05585262 J)
        close(
            rel(Power::from_si(1.0).as_ip(), 3600.0 / 1_055.055_852_62),
            -1.4e-7,
            1e-8,
        );
        close(
            rel(
                HeatFluxDensity::from_si(1.0).as_ip(),
                3600.0 / 1_055.055_852_62 * 0.3048 * 0.3048,
            ),
            -1.4e-7,
            1e-8,
        );
        // met: 18.4 Btu/(h·ft²) as a metabolic rate, 18.433 as the same heat flux
        close(MetabolicRate::from_met(1.0).as_ip(), 18.4, 1e-12);
        close(HeatFluxDensity::from_si(58.15).as_ip(), 18.433, 1e-3);
        // factors built on the Btu: -1.4e-7 relative, like W -> Btu/h
        let btu_j = 1_055.055_852_62;
        let ft2 = 0.3048 * 0.3048;
        close(
            rel(SpecificEnthalpy::from_si(1.0).as_ip(), 1.0 / 2326.0),
            -1.4e-7,
            1e-8,
        );
        close(
            rel(HeatPerArea::from_si(1.0).as_ip(), ft2 / btu_j),
            -1.4e-7,
            1e-8,
        );
        close(
            rel(PowerRate::from_si(1.0).as_ip(), 3600.0 / btu_j),
            -1.4e-7,
            1e-8,
        );
        close(
            rel(
                ThermalResistance::from_si(1.0).as_ip(),
                1.8 * btu_j / 3600.0 / ft2,
            ),
            1.4e-7,
            1e-8,
        );
        // m²·kPa/W -> ft²·h·psi/Btu: psi and Btu combined, -2.5e-7 relative
        close(
            rel(
                EvaporativeResistance::from_si(1.0).as_ip(),
                1000.0 / 6_894.757_293_168 * btu_j / 3600.0 / ft2,
            ),
            -2.5e-7,
            1e-8,
        );
        // exact factors
        close(Length::from_ip(1.0).as_si(), 0.3048, 1e-16);
        close(Mass::from_ip(1.0).as_si(), 0.453_592_37, 1e-16);
        close(TemperatureDelta::from_si(1.0).as_ip(), 1.8, 1e-16);
        let exact = |got: f64, want: f64| close(rel(got, want), 0.0, 4e-16);
        exact(MassFluxDensity::from_ip(1.0).as_si(), 453.592_37 / ft2);
        exact(VolumeFluxDensity::from_ip(1.0).as_si(), 3.785_411_784 / ft2);
        exact(VolumeFlow::from_ip(1.0).as_si(), 3.785_411_784);
        exact(MassFlow::from_ip(1.0).as_si(), 453.592_37 / 3600.0);
        exact(HumidityRatio::from_si(0.01).as_ip(), 0.01);
        exact(HumidityRatio::from_si(0.01).as_grams_per_kilogram(), 10.0);
        exact(HumidityRatio::from_si(0.01).as_grains_per_pound(), 70.0);
        exact(
            HeatPerArea::from_si(120.0).as_watt_minutes_per_square_meter(),
            2.0,
        );
        exact(
            SweatingCoefficient::from_ip(1.0).as_si(),
            453.592_37 / ft2 * 1.8,
        );
        exact(
            VasodilationCoefficient::from_ip(1.0).as_si(),
            3.785_411_784 / ft2 * 1.8,
        );
        exact(VasoconstrictionCoefficient::from_ip(1.0).as_si(), 1.8);
    }

    #[test]
    fn units_dispatch() {
        let t = Temperature::from_units(77.0, Units::IP);
        assert!((t.in_units(Units::SI) - 25.0).abs() < 1e-9);
        assert_eq!(Temperature::unit(Units::IP), "°F");
        assert_eq!(celsius_in_units(25.0, Units::IP), 77.0);
        assert_eq!(celsius_in_units(25.0, Units::SI), 25.0);
    }
}
