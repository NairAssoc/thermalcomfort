//! Example using the type-safe API with the measurements crate

use thermalcomfort::models::pmv_typed::{pmv_ppd_iso_typed, pmv_ppd_ashrae_typed};
use thermalcomfort::utilities::v_relative;
use measurements::{Temperature, Speed};

fn main() {
    println!("=== Type-Safe Thermal Comfort API Example ===\n");

    // Example 1: Using Fahrenheit and km/h (automatically converted)
    println!("Example 1: International Units");
    let tdb_f = Temperature::from_fahrenheit(77.0);
    let tr_c = Temperature::from_celsius(25.0);
    let v_kmh = Speed::from_kilometers_per_hour(0.36);
    let rh = 50.0;
    let met = 1.4;
    let clo = 0.5;

    println!("  Temperature (F): {:.1}°F = {:.1}°C",
             tdb_f.as_fahrenheit(), tdb_f.as_celsius());
    println!("  Air speed: {:.2} km/h = {:.2} m/s",
             v_kmh.as_kilometers_per_hour(), v_kmh.as_meters_per_second());

    // Calculate relative air speed
    let vr_ms = v_relative(v_kmh.as_meters_per_second(), met);
    let vr = Speed::from_meters_per_second(vr_ms);

    let result = pmv_ppd_iso_typed(tdb_f, tr_c, vr, rh, met, clo, Default::default());
    println!("  PMV: {:.2}", result.pmv);
    println!("  PPD: {:.1}%", result.ppd);
    println!("  Thermal Sensation: {:?}\n", result.tsv);

    // Example 2: Comparing different units
    println!("Example 2: Unit Conversion Verification");
    let temp_c = Temperature::from_celsius(20.0);
    let temp_f = Temperature::from_fahrenheit(68.0);

    println!("  20°C = {:.1}°F", temp_c.as_fahrenheit());
    println!("  68°F = {:.1}°C", temp_f.as_celsius());

    // Both should give same result
    let vr = Speed::from_meters_per_second(0.1);
    let result_c = pmv_ppd_iso_typed(temp_c, temp_c, vr, 50.0, 1.2, 1.0, Default::default());
    let result_f = pmv_ppd_iso_typed(temp_f, temp_f, vr, 50.0, 1.2, 1.0, Default::default());

    println!("  PMV (using Celsius): {:.2}", result_c.pmv);
    println!("  PMV (using Fahrenheit): {:.2}", result_f.pmv);
    println!("  Difference: {:.4} (should be ~0)\n", (result_c.pmv - result_f.pmv).abs());

    // Example 3: ASHRAE calculation with typed API
    println!("Example 3: ASHRAE 55 with Type Safety");
    let tdb = Temperature::from_celsius(25.0);
    let tr = Temperature::from_celsius(25.0);
    let v = Speed::from_meters_per_second(0.1);
    let rh = 50.0;
    let met = 1.2;
    let clo = 0.5;

    let result = pmv_ppd_ashrae_typed(tdb, tr, v, rh, met, clo, Default::default());
    println!("  PMV (ASHRAE): {:.2}", result.pmv);
    println!("  PPD: {:.1}%", result.ppd);
    println!("  Thermal Sensation: {:?}\n", result.tsv);

    // Example 4: Demonstrating type safety
    println!("Example 4: Type Safety Benefits");
    println!("  With typed API, the compiler ensures:");
    println!("  ✓ Temperature is Temperature, not confused with Speed");
    println!("  ✓ Speed is Speed, not confused with Temperature");
    println!("  ✓ Units are automatically converted");
    println!("  ✓ No risk of passing Fahrenheit where Celsius is expected");
    println!("  ✓ Clear self-documenting code");
}
