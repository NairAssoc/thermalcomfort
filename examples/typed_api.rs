//! Working in whatever units you have, and letting the types do the conversion.
//!
//! This used to demonstrate a separate "typed API" sitting alongside an untyped one. There
//! is no longer such a split — every model takes typed quantities and named inputs — so what
//! is left worth showing is the part that was always the point: you hand the library the
//! units you actually measured in, and comparisons that would be unit errors elsewhere are
//! either impossible to write or converted for you.

use thermalcomfort::models::pmv::PmvPpdInputs;
use thermalcomfort::models::{pmv_ppd_ashrae, pmv_ppd_iso};
use thermalcomfort::utilities::v_relative;
use thermalcomfort::{ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature};

fn main() {
    println!("=== Units and typed quantities ===\n");

    // 1. Mixed units in, no manual conversion.
    println!("1. Fahrenheit and km/h, converted by the types");
    let tdb = Temperature::from_fahrenheit(77.0);
    let tr = Temperature::from_celsius(25.0);
    let v = Speed::from_kilometers_per_hour(0.36);
    let met = MetabolicRate::from_met(1.4);
    let clo = ClothingInsulation::from_clo(0.5);

    println!(
        "   {:.1}°F = {:.1}°C, {:.2} km/h = {:.2} m/s",
        tdb.as_fahrenheit(),
        tdb.as_celsius(),
        v.as_kilometers_per_hour(),
        v.as_meters_per_second()
    );

    let result = pmv_ppd_iso(
        PmvPpdInputs {
            tdb,
            tr,
            vr: v_relative(v, met),
            rh: Humidity::from_percent(50.0),
            met,
            clo,
        },
        Default::default(),
    );
    println!("   PMV {:.2}, PPD {:.1}%\n", result.pmv, result.ppd);

    // 2. The same temperature written two ways gives the same answer.
    println!("2. 20°C and 68°F are the same input");
    let in_celsius = Temperature::from_celsius(20.0);
    let in_fahrenheit = Temperature::from_fahrenheit(68.0);

    let pmv_at = |t: Temperature| {
        pmv_ppd_iso(
            PmvPpdInputs {
                tdb: t,
                tr: t,
                vr: Speed::from_meters_per_second(0.1),
                rh: Humidity::from_percent(50.0),
                met: MetabolicRate::from_met(1.2),
                clo: ClothingInsulation::from_clo(1.0),
            },
            Default::default(),
        )
        .pmv
    };

    println!("   from_celsius(20.0)    -> PMV {:.2}", pmv_at(in_celsius));
    println!(
        "   from_fahrenheit(68.0) -> PMV {:.2}\n",
        pmv_at(in_fahrenheit)
    );

    // 3. Named inputs, so a mis-ordered call does not compile rather than
    //    silently computing the wrong thing.
    println!("3. ASHRAE 55, with every input named");
    let result = pmv_ppd_ashrae(
        PmvPpdInputs {
            tdb: Temperature::from_celsius(25.0),
            tr: Temperature::from_celsius(25.0),
            vr: Speed::from_meters_per_second(0.1),
            rh: Humidity::from_percent(50.0),
            met: MetabolicRate::from_met(1.2),
            clo: ClothingInsulation::from_clo(0.5),
        },
        Default::default(),
    );
    println!("   PMV {:.2}, PPD {:.1}%", result.pmv, result.ppd);
    println!("   Sensation: {:?}\n", result.tsv);

    println!("What the types rule out:");
    println!("  - a Speed where a Temperature belongs: will not compile");
    println!("  - °F read as °C: converted, not mistaken");
    println!("  - tdb and tr transposed: they are named, not positional");
    println!("  - a temperature *difference* used as an absolute reading:");
    println!("    TemperatureDelta is a distinct type from Temperature");
}
