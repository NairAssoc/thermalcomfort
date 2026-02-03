//! Heat and cold stress indices example
//!
//! Demonstrates various thermal stress indices for extreme conditions.

use thermalcomfort::models::{
    heat_index_rothfusz, humidex, thi, discomfort_index,
    wci, wind_chill_temperature
};
use measurements::{Temperature, Speed};

fn main() {
    println!("=== Thermal Stress Indices ===\n");

    // HEAT STRESS INDICES
    println!("--- Heat Stress Assessment ---\n");

    let hot_temp = 35.0;  // °C
    let hot_rh = 60.0;    // %

    println!("Conditions: {:.0}°C, {:.0}% RH\n", hot_temp, hot_rh);

    // Heat Index (Rothfusz)
    let hi = heat_index_rothfusz(
        Temperature::from_celsius(hot_temp),
        hot_rh,
        true,
        true
    );
    println!("Heat Index (Rothfusz):");
    println!("  {:.1}°C", hi);
    if hi < 27.0 {
        println!("  Caution: Fatigue possible with prolonged exposure");
    } else if hi < 32.0 {
        println!("  Extreme caution: Heat exhaustion possible");
    } else if hi < 41.0 {
        println!("  Danger: Heat cramps and exhaustion likely");
    } else {
        println!("  Extreme danger: Heat stroke imminent");
    }

    // Humidex
    let humidex_val = humidex(
        Temperature::from_celsius(hot_temp),
        hot_rh,
        true
    );
    println!("\nHumidex (Canadian):");
    println!("  {:.0}", humidex_val);
    if humidex_val < 30.0 {
        println!("  No discomfort");
    } else if humidex_val < 40.0 {
        println!("  Some discomfort");
    } else if humidex_val < 45.0 {
        println!("  Great discomfort; avoid exertion");
    } else {
        println!("  Dangerous; heat stroke possible");
    }

    // Temperature-Humidity Index (THI)
    let thi_val = thi(
        Temperature::from_celsius(hot_temp),
        hot_rh,
        true
    );
    println!("\nTemperature-Humidity Index:");
    println!("  {:.1}", thi_val);
    if thi_val < 70.0 {
        println!("  Comfortable");
    } else if thi_val < 75.0 {
        println!("  Slightly uncomfortable");
    } else if thi_val < 80.0 {
        println!("  Uncomfortable");
    } else {
        println!("  Very uncomfortable");
    }

    // Discomfort Index
    let di = discomfort_index(
        Temperature::from_celsius(hot_temp),
        hot_rh
    );
    println!("\nDiscomfort Index:");
    println!("  {:.1}", di);
    if di < 21.0 {
        println!("  No discomfort");
    } else if di < 24.0 {
        println!("  Less than 50% feel discomfort");
    } else if di < 27.0 {
        println!("  More than 50% feel discomfort");
    } else if di < 29.0 {
        println!("  Most people uncomfortable");
    } else {
        println!("  Everyone feels severe stress");
    }

    // COLD STRESS INDICES
    println!("\n\n--- Cold Stress Assessment ---\n");

    let cold_temp = -10.0;  // °C
    let wind_speed_kmh = 20.0;  // km/h
    let wind_speed_ms = wind_speed_kmh / 3.6;  // convert to m/s

    println!("Conditions: {:.0}°C, {:.0} km/h wind\n", cold_temp, wind_speed_kmh);

    // Wind Chill Temperature
    let wct = wind_chill_temperature(
        Temperature::from_celsius(cold_temp),
        Speed::from_kilometers_per_hour(wind_speed_kmh),
        true
    );
    println!("Wind Chill Temperature:");
    println!("  {:.1}°C (feels like)", wct);
    if wct > -10.0 {
        println!("  Low risk");
    } else if wct > -28.0 {
        println!("  Low risk of frostbite");
    } else if wct > -40.0 {
        println!("  Moderate risk: frostbite in 10-30 minutes");
    } else if wct > -48.0 {
        println!("  High risk: frostbite in 5-10 minutes");
    } else {
        println!("  Extreme risk: frostbite in 2-5 minutes");
    }

    // Wind Chill Index
    let wci_val = wci(
        Temperature::from_celsius(cold_temp),
        Speed::from_meters_per_second(wind_speed_ms),
        true
    );
    println!("\nWind Chill Index:");
    println!("  {:.0} W/m²", wci_val);
    if wci_val < 600.0 {
        println!("  Comfortable with proper clothing");
    } else if wci_val < 1200.0 {
        println!("  Uncomfortably cold");
    } else if wci_val < 1500.0 {
        println!("  Very cold; exposed skin freezes");
    } else if wci_val < 1800.0 {
        println!("  Bitterly cold; frostbite risk");
    } else {
        println!("  Extremely cold; outdoor activity dangerous");
    }

    // COMPARISON OF METHODS
    println!("\n\n--- Comparison Example ---");
    println!("For 30°C with varying humidity:\n");

    for rh in [30.0, 50.0, 70.0, 90.0] {
        let hi = heat_index_rothfusz(
            Temperature::from_celsius(30.0),
            rh,
            true,
            true
        );
        let hum = humidex(
            Temperature::from_celsius(30.0),
            rh,
            true
        );

        println!("  RH {:.0}%: HI = {:.1}°C, Humidex = {:.0}",
                 rh, hi, hum);
    }

    println!("\nNote: Different indices emphasize different aspects:");
    println!("  • Heat Index: Based on Steadman's model, widely used in US");
    println!("  • Humidex: Canadian system, simpler calculation");
    println!("  • THI: Traditional index, originally for animal comfort");
    println!("  • Wind Chill: Standardized across North America");
}
