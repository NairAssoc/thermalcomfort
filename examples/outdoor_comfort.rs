//! Outdoor thermal comfort example
//!
//! Demonstrates UTCI (Universal Thermal Climate Index) and WBGT (Wet Bulb Globe Temperature)
//! for assessing outdoor thermal conditions and heat stress.

use thermalcomfort::models::{utci, wbgt};
use thermalcomfort::psychrometrics::wet_bulb_temperature;
use thermalcomfort::{Temperature, Speed, Humidity};

fn main() {
    println!("=== Outdoor Thermal Comfort Assessment ===\n");

    // Example 1: Moderate summer day
    println!("--- Example 1: Moderate Summer Day ---");
    let tdb1 = Temperature::from_celsius(28.0);  // dry bulb temperature
    let tr1 = Temperature::from_celsius(32.0);   // mean radiant temperature (higher due to solar radiation)
    let v1 = Speed::from_meters_per_second(1.5); // wind speed
    let rh1 = Humidity::from_percent(50.0);      // relative humidity

    println!("Conditions:");
    println!("  Dry bulb temperature: {:.1}°C", tdb1.as_celsius());
    println!("  Mean radiant temp:    {:.1}°C (elevated by sun)", tr1.as_celsius());
    println!("  Wind speed:           {:.1} m/s", v1.as_meters_per_second());
    println!("  Relative humidity:    {:.0}%\n", rh1.as_percent());

    // UTCI calculation
    let utci_result1 = utci(
        tdb1,
        tr1,
        v1,
        rh1,
        Default::default()
    );

    println!("UTCI Assessment:");
    println!("  UTCI: {:.1}°C", utci_result1.utci);
    println!("  Thermal stress: {}", utci_result1.stress_category.as_str());

    // WBGT calculation (outdoor with solar load)
    let twb1 = wet_bulb_temperature(tdb1, rh1);
    let tg1 = Temperature::from_celsius(35.0); // globe temperature (elevated by solar radiation)

    let wbgt_result1 = wbgt(
        twb1,
        tg1,
        Some(tdb1),
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
    let tdb2 = Temperature::from_celsius(35.0);
    let tr2 = Temperature::from_celsius(40.0);  // very high radiant temperature
    let v2 = Speed::from_meters_per_second(0.5);    // low wind
    let rh2 = Humidity::from_percent(70.0);  // high humidity

    println!("Conditions:");
    println!("  Dry bulb temperature: {:.1}°C", tdb2.as_celsius());
    println!("  Mean radiant temp:    {:.1}°C", tr2.as_celsius());
    println!("  Wind speed:           {:.1} m/s", v2.as_meters_per_second());
    println!("  Relative humidity:    {:.0}%\n", rh2.as_percent());

    let utci_result2 = utci(
        tdb2,
        tr2,
        v2,
        rh2,
        Default::default()
    );

    println!("UTCI Assessment:");
    println!("  UTCI: {:.1}°C", utci_result2.utci);
    println!("  Thermal stress: {}", utci_result2.stress_category.as_str());

    let twb2 = wet_bulb_temperature(tdb2, rh2);
    let tg2 = Temperature::from_celsius(42.0);

    let wbgt_result2 = wbgt(
        twb2,
        tg2,
        Some(tdb2),
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
    let tdb3 = Temperature::from_celsius(-5.0);
    let tr3 = Temperature::from_celsius(-8.0);  // lower due to cold surfaces
    let v3 = Speed::from_meters_per_second(3.0);    // wind chill factor
    let rh3 = Humidity::from_percent(80.0);

    println!("Conditions:");
    println!("  Dry bulb temperature: {:.1}°C", tdb3.as_celsius());
    println!("  Mean radiant temp:    {:.1}°C", tr3.as_celsius());
    println!("  Wind speed:           {:.1} m/s", v3.as_meters_per_second());
    println!("  Relative humidity:    {:.0}%\n", rh3.as_percent());

    let utci_result3 = utci(
        tdb3,
        tr3,
        v3,
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
