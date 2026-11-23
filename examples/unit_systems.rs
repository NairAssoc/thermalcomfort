//! SI and IP (inch-pound) units example
//!
//! Every quantity is typed, so a value means the same thing whichever unit system it was
//! built in. `UnitSystem` gives each type `from_si`/`as_si` and `from_ip`/`as_ip`, so a
//! caller working entirely in SI, or entirely in IP (inch-pound: °F, ft, ft/s, psi, Btu),
//! never picks a unit per quantity.

use thermalcomfort::models::{
    AdaptiveInputs, AdaptiveOptions, CoolingEffectInputs, CoolingEffectOptions,
    GaggeTwoNodesInputs, adaptive_ashrae, cooling_effect, two_nodes_gagge,
};
use thermalcomfort::{
    AirPermeability, Area, ClothingInsulation, HeatFluxDensity, Humidity, Length, Mass,
    MetabolicRate, Power, Pressure, Speed, Temperature, TemperatureDelta, UnitSystem, Units,
};

/// Print a quantity in both systems, with the symbol of each unit.
fn both<T: UnitSystem>(label: &str, value: T) {
    println!(
        "  {label:<22} {:>10.3} {:<12} = {:>10.3} {}",
        value.as_si(),
        T::SI_UNIT,
        value.as_ip(),
        T::IP_UNIT
    );
}

fn main() {
    println!("=== SI and IP (inch-pound) units ===\n");

    // --- Every type, both ways ---
    println!("--- One value per type, read in SI and in IP ---");
    both("Temperature", Temperature::from_si(25.0));
    both("Temperature difference", TemperatureDelta::from_si(2.0));
    both("Air speed", Speed::from_si(0.8));
    both("Height", Length::from_si(1.75));
    both("Body surface area", Area::from_si(1.8));
    both("Body mass", Mass::from_si(70.0));
    both("Atmospheric pressure", Pressure::from_si(101_325.0));
    both("Power", Power::from_si(100.0));
    both("Heat flux", HeatFluxDensity::from_si(58.15));
    both("Metabolic rate", MetabolicRate::from_met(1.2));
    both("Clothing insulation", ClothingInsulation::from_clo(0.5));
    both("Air permeability", AirPermeability::from_si(50.0));

    // --- The same office, described by an SI user and by an IP user ---
    println!("\n--- The same conditions, entered and read in each system ---");
    let si = (
        Temperature::from_si(30.0),
        Speed::from_si(0.8),
        Temperature::from_si(24.0),
    );
    let ip = (
        Temperature::from_ip(86.0),
        Speed::from_ip(0.8 * 3.281),
        Temperature::from_ip(75.2),
    );
    for (name, units, (t, v, t_rm)) in [("SI", Units::SI, si), ("IP", Units::IP, ip)] {
        let t_unit = Temperature::unit(units);
        println!(
            "\n  {name} user: {:.1} {t_unit}, {:.2} {}",
            t.in_units(units),
            v.in_units(units),
            Speed::unit(units)
        );

        // `units` selects the unit pythermalcomfort rounds the result in; reading the
        // result in the same system gives upstream's number exactly
        let ce = cooling_effect(
            CoolingEffectInputs {
                tdb: t,
                tr: t,
                vr: v,
                rh: Humidity::from_percent(50.0),
                met: MetabolicRate::from_met(1.2),
                clo: ClothingInsulation::from_clo(0.5),
            },
            CoolingEffectOptions {
                units,
                ..Default::default()
            },
        );
        println!(
            "    Cooling effect:      {:.2} {}",
            ce.in_units(units),
            TemperatureDelta::unit(units)
        );

        let adaptive = adaptive_ashrae(
            AdaptiveInputs {
                tdb: t,
                tr: t,
                t_running_mean: t_rm,
                v,
            },
            AdaptiveOptions {
                units,
                ..Default::default()
            },
        );
        println!(
            "    Adaptive comfort:    {:.1} {t_unit} (80 % band {:.1}-{:.1} {t_unit})",
            adaptive.tmp_cmf.in_units(units),
            adaptive.tmp_cmf_80_low.in_units(units),
            adaptive.tmp_cmf_80_up.in_units(units)
        );

        let gagge = two_nodes_gagge(
            GaggeTwoNodesInputs {
                tdb: t,
                tr: t,
                v,
                rh: Humidity::from_percent(50.0),
                met: MetabolicRate::from_met(1.2),
                clo: ClothingInsulation::from_clo(0.5),
            },
            Default::default(),
        );
        println!(
            "    SET:                 {:.1} {t_unit}, evaporative loss {:.1} {}",
            gagge.set.in_units(units),
            gagge.e_skin.in_units(units),
            HeatFluxDensity::unit(units)
        );
    }

    // Either result can be read in either system: an IP result read with as_si() is the
    // same physical difference in °C, to within one rounding step in °F
    let ce_ip = cooling_effect(
        CoolingEffectInputs {
            tdb: Temperature::from_ip(86.0),
            tr: Temperature::from_ip(86.0),
            vr: Speed::from_ip(0.8 * 3.281),
            rh: Humidity::from_percent(50.0),
            met: MetabolicRate::from_met(1.2),
            clo: ClothingInsulation::from_clo(0.5),
        },
        CoolingEffectOptions {
            units: Units::IP,
            ..Default::default()
        },
    );
    println!(
        "\n  The IP cooling effect, read both ways: {:.2} °F = {:.3} °C",
        ce_ip.as_ip(),
        ce_ip.as_si()
    );
}
