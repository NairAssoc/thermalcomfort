//! Outdoor thermal comfort example
//!
//! Demonstrates UTCI (Universal Thermal Climate Index) and WBGT (Wet Bulb Globe Temperature)
//! for assessing outdoor thermal conditions and heat stress.

use thermalcomfort::models::{utci, wbgt};
use thermalcomfort::psychrometrics::wet_bulb_temperature;
use measurements::{Temperature, Speed};

fn main() {
    println!("=== Outdoor Thermal Comfort Assessment ===\n");

    // Example 1: Moderate summer day
    println!("--- Example 1: Moderate Summer Day ---");
    let tdb1 = 28.0;  // dry bulb temperature [°C]
    let tr1 = 32.0;   // mean radiant temperature (higher due to solar radiation) [°C]
    let v1 = 1.5;     // wind speed [m/s]
    let rh1 = 50.0;   // relative humidity [%]

    println!("Conditions:");
    println!("  Dry bulb temperature: {:.1}°C", tdb1);
    println!("  Mean radiant temp:    {:.1}°C (elevated by sun)", tr1);
    println!("  Wind speed:           {:.1} m/s", v1);
    println!("  Relative humidity:    {:.0}%\n", rh1);

    // UTCI calculation
    let utci_result1 = utci(
        Temperature::from_celsius(tdb1),
        Temperature::from_celsius(tr1),
        Speed::from_meters_per_second(v1),
        rh1,
        Default::default()
    );

    println!("UTCI Assessment:");
    println!("  UTCI: {:.1}°C", utci_result1.utci);
    println!("  Thermal stress: {}", utci_result1.stress_category.as_str());

    // WBGT calculation (outdoor with solar load)
    let twb1 = wet_bulb_temperature(tdb1, rh1);
    let tg1 = 35.0; // globe temperature (elevated by solar radiation)

    let wbgt_result1 = wbgt(
        Temperature::from_celsius(twb1),
        Temperature::from_celsius(tg1),
        Some(Temperature::from_celsius(tdb1)),
        Default::default()
    );

    println!("\nWBGT Heat Stress Assessment:");
    println!("  WBGT: {:.1}°C", wbgt_result1);

    if wbgt_result1 < 28.0 {
        println!("  Risk level: Low - Normal activities OK");
    } else if wbgt_result1 < 31.0 {
        println!("  Risk level: Moderate - Caution for prolonged exposure");
    } else if wbgt_result1 < 33.0 {
        println!("  Risk level: High - Limit strenuous activities");
    } else {
        println!("  Risk level: Extreme - Avoid outdoor work");
    }

    // Example 2: Hot and humid day
    println!("\n\n--- Example 2: Hot and Humid Day ---");
    let tdb2 = 35.0;
    let tr2 = 40.0;  // very high radiant temperature
    let v2 = 0.5;    // low wind
    let rh2 = 70.0;  // high humidity

    println!("Conditions:");
    println!("  Dry bulb temperature: {:.1}°C", tdb2);
    println!("  Mean radiant temp:    {:.1}°C", tr2);
    println!("  Wind speed:           {:.1} m/s", v2);
    println!("  Relative humidity:    {:.0}%\n", rh2);

    let utci_result2 = utci(
        Temperature::from_celsius(tdb2),
        Temperature::from_celsius(tr2),
        Speed::from_meters_per_second(v2),
        rh2,
        Default::default()
    );

    println!("UTCI Assessment:");
    println!("  UTCI: {:.1}°C", utci_result2.utci);
    println!("  Thermal stress: {}", utci_result2.stress_category.as_str());

    let twb2 = wet_bulb_temperature(tdb2, rh2);
    let tg2 = 42.0;

    let wbgt_result2 = wbgt(
        Temperature::from_celsius(twb2),
        Temperature::from_celsius(tg2),
        Some(Temperature::from_celsius(tdb2)),
        Default::default()
    );

    println!("\nWBGT Heat Stress Assessment:");
    println!("  WBGT: {:.1}°C", wbgt_result2);

    if wbgt_result2 >= 33.0 {
        println!("  ⚠ EXTREME HEAT STRESS");
        println!("  Recommendations:");
        println!("    • Avoid outdoor physical work");
        println!("    • Seek air-conditioned shelter");
        println!("    • Stay hydrated");
        println!("    • Check on vulnerable individuals");
    }

    // Example 3: Cold winter day
    println!("\n\n--- Example 3: Cold Winter Day ---");
    let tdb3 = -5.0;
    let tr3 = -8.0;  // lower due to cold surfaces
    let v3 = 3.0;    // wind chill factor
    let rh3 = 80.0;

    println!("Conditions:");
    println!("  Dry bulb temperature: {:.1}°C", tdb3);
    println!("  Mean radiant temp:    {:.1}°C", tr3);
    println!("  Wind speed:           {:.1} m/s", v3);
    println!("  Relative humidity:    {:.0}%\n", rh3);

    let utci_result3 = utci(
        Temperature::from_celsius(tdb3),
        Temperature::from_celsius(tr3),
        Speed::from_meters_per_second(v3),
        rh3,
        Default::default()
    );

    println!("UTCI Assessment:");
    println!("  UTCI: {:.1}°C", utci_result3.utci);
    println!("  Thermal stress: {}", utci_result3.stress_category.as_str());

    println!("\n--- Summary ---");
    println!("UTCI provides a comprehensive measure of outdoor thermal comfort");
    println!("accounting for temperature, radiation, wind, and humidity.");
    println!("\nWBGT is specifically designed for heat stress assessment and is");
    println!("widely used in occupational health and sports medicine.");
}
