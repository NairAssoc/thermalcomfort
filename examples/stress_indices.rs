//! Heat and cold stress indices example
//!
//! Demonstrates various thermal stress indices for extreme conditions.

use thermalcomfort::models::{
    discomfort_index, heat_index_rothfusz, humidex, thi, wci, wind_chill_temperature,
};
use thermalcomfort::{Humidity, Speed, Temperature};

fn main() {
    println!("=== Thermal Stress Indices ===\n");

    // HEAT STRESS INDICES
    println!("--- Heat Stress Assessment ---\n");

    let hot_temp = Temperature::from_celsius(35.0);
    let hot_rh = Humidity::from_percent(60.0);

    println!(
        "Conditions: {:.0}°C, {:.0}% RH\n",
        hot_temp.as_celsius(),
        hot_rh.as_percent()
    );

    // Heat Index (Rothfusz)
    let hi_result = heat_index_rothfusz(hot_temp, hot_rh, true, true);
    println!("Heat Index (Rothfusz):");
    println!("  {:.1}°C", hi_result.hi);
    if let Some(cat) = hi_result.stress_category {
        println!("  {}", cat.as_str());
    } else {
        println!("  (outside applicability range)");
    }

    // Humidex
    let humidex_result = humidex(hot_temp, hot_rh, true);
    println!("\nHumidex (Canadian):");
    println!("  {:.0}", humidex_result.humidex);
    println!("  {}", humidex_result.discomfort.as_str());

    // Temperature-Humidity Index (THI)
    let thi_val = thi(hot_temp, hot_rh, true);
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
    let di_result = discomfort_index(hot_temp, hot_rh);
    println!("\nDiscomfort Index:");
    println!("  {:.1}", di_result.di);
    println!("  {}", di_result.discomfort_condition.as_str());

    // COLD STRESS INDICES
    println!("\n\n--- Cold Stress Assessment ---\n");

    let cold_temp = Temperature::from_celsius(-10.0);
    let wind_speed = Speed::from_kilometers_per_hour(20.0);

    println!(
        "Conditions: {:.0}°C, {:.0} km/h wind\n",
        cold_temp.as_celsius(),
        wind_speed.as_kilometers_per_hour()
    );

    // Wind Chill Temperature
    let wct = wind_chill_temperature(cold_temp, wind_speed, true);
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
    let wci_val = wci(cold_temp, wind_speed, true);
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

    let temp_30 = Temperature::from_celsius(30.0);
    for rh_val in [30.0, 50.0, 70.0, 90.0] {
        let rh = Humidity::from_percent(rh_val);
        let hi = heat_index_rothfusz(temp_30, rh, true, true);
        let hum = humidex(temp_30, rh, true);

        println!(
            "  RH {:.0}%: HI = {:.1}°C, Humidex = {:.0}",
            rh_val, hi.hi, hum.humidex
        );
    }

    println!("\nNote: Different indices emphasize different aspects:");
    println!("  • Heat Index: Based on Steadman's model, widely used in US");
    println!("  • Humidex: Canadian system, simpler calculation");
    println!("  • THI: Traditional index, originally for animal comfort");
    println!("  • Wind Chill: Standardized across North America");
}
