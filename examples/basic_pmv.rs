//! Basic example of PMV/PPD calculation

use thermalcomfort::models::pmv_ppd_iso;
use thermalcomfort::utilities::v_relative;
use measurements::{Temperature, Speed};

fn main() {
    println!("=== Thermal Comfort PMV/PPD Example ===\n");

    // Environmental parameters
    let tdb = 25.0;  // dry bulb temperature [°C]
    let tr = 25.0;   // mean radiant temperature [°C]
    let rh = 50.0;   // relative humidity [%]
    let v = 0.1;     // air speed [m/s]
    let met = 1.4;   // metabolic rate [met]
    let clo = 0.5;   // clothing insulation [clo]

    println!("Environmental Conditions:");
    println!("  Temperature (dry bulb): {:.1}°C", tdb);
    println!("  Mean radiant temp:      {:.1}°C", tr);
    println!("  Relative humidity:      {:.0}%", rh);
    println!("  Air speed:              {:.1} m/s", v);
    println!("  Metabolic rate:         {:.1} met", met);
    println!("  Clothing insulation:    {:.1} clo", clo);

    // Calculate relative air speed (accounts for body movement)
    let vr = v_relative(v, met);
    println!("\n  Relative air speed:     {:.2} m/s", vr);

    // Calculate PMV and PPD using measurement types
    let result = pmv_ppd_iso(
        Temperature::from_celsius(tdb),
        Temperature::from_celsius(tr),
        Speed::from_meters_per_second(vr),
        rh,
        met,
        clo,
        Default::default()
    );

    println!("\nThermal Comfort Results:");
    println!("  PMV (Predicted Mean Vote):              {:.2}", result.pmv);
    println!("  PPD (Predicted % Dissatisfied):         {:.1}%", result.ppd);
    println!("  Thermal Sensation:                      {:?}", result.tsv);

    // Interpretation
    println!("\nInterpretation:");
    if result.pmv.abs() < 0.5 {
        println!("  The environment is thermally NEUTRAL and comfortable.");
    } else if result.pmv > 0.0 {
        println!("  The environment is WARM to HOT.");
    } else {
        println!("  The environment is COOL to COLD.");
    }

    if result.ppd < 10.0 {
        println!("  Less than 10% of people are expected to be dissatisfied.");
        println!("  This meets typical comfort standards.");
    } else {
        println!("  More than 10% of people are expected to be dissatisfied.");
        println!("  Consider adjusting environmental conditions.");
    }
}
