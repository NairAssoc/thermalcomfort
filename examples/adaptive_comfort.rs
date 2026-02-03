//! Adaptive comfort model example
//!
//! Demonstrates the use of adaptive comfort models (ASHRAE 55 and EN 16798-1)
//! for naturally ventilated buildings.

use thermalcomfort::models::{adaptive_ashrae, adaptive_en};
use thermalcomfort::utilities::running_mean_outdoor_temperature;
use thermalcomfort::{Temperature, Speed};

fn main() {
    println!("=== Adaptive Thermal Comfort Models ===\n");

    // Environmental conditions
    let tdb = Temperature::from_celsius(25.0);  // indoor operative temperature
    let tr = Temperature::from_celsius(25.0);   // mean radiant temperature
    let v = Speed::from_meters_per_second(0.2); // air speed

    // Historical outdoor temperatures for running mean calculation
    let outdoor_temps = vec![
        Temperature::from_celsius(22.0),
        Temperature::from_celsius(21.5),
        Temperature::from_celsius(20.0),
        Temperature::from_celsius(19.5),
        Temperature::from_celsius(18.0),
        Temperature::from_celsius(17.5),
        Temperature::from_celsius(17.0),
    ];
    let t_running_mean = running_mean_outdoor_temperature(&outdoor_temps, 0.8);

    println!("Indoor Conditions:");
    println!("  Operative temperature: {:.1}°C", tdb.as_celsius());
    println!("  Air speed:             {:.1} m/s", v.as_meters_per_second());
    println!("  Running mean outdoor:  {:.1}°C\n", t_running_mean.as_celsius());

    // ASHRAE 55 Adaptive Model
    println!("--- ASHRAE 55 Adaptive Model ---");
    let ashrae_result = adaptive_ashrae(
        tdb,
        tr,
        t_running_mean,
        v,
        Default::default()
    );

    println!("  Comfort temperature:    {:.1}°C", ashrae_result.tmp_cmf);
    println!("  80% acceptability:      {:.1}°C to {:.1}°C",
             ashrae_result.tmp_cmf_80_low, ashrae_result.tmp_cmf_80_up);
    println!("  90% acceptability:      {:.1}°C to {:.1}°C",
             ashrae_result.tmp_cmf_90_low, ashrae_result.tmp_cmf_90_up);

    if ashrae_result.acceptability_80 {
        println!("  ✓ Acceptable for 80% of occupants");
    } else {
        println!("  ✗ Not acceptable for 80% of occupants");
    }

    if ashrae_result.acceptability_90 {
        println!("  ✓ Acceptable for 90% of occupants");
    } else {
        println!("  ✗ Not acceptable for 90% of occupants");
    }

    // EN 16798-1 Adaptive Model
    println!("\n--- EN 16798-1 Adaptive Model ---");
    let en_result = adaptive_en(
        tdb,
        tr,
        t_running_mean,
        v,
        Default::default()
    );

    println!("  Comfort temperature:    {:.1}°C", en_result.tmp_cmf);
    println!("  Category I (90%):       {:.1}°C to {:.1}°C",
             en_result.tmp_cmf_cat_i_low, en_result.tmp_cmf_cat_i_up);
    println!("  Category II (80%):      {:.1}°C to {:.1}°C",
             en_result.tmp_cmf_cat_ii_low, en_result.tmp_cmf_cat_ii_up);
    println!("  Category III (65%):     {:.1}°C to {:.1}°C",
             en_result.tmp_cmf_cat_iii_low, en_result.tmp_cmf_cat_iii_up);

    print!("\n  Comfort category: ");
    if en_result.acceptability_cat_i {
        println!("Category I (highest comfort)");
    } else if en_result.acceptability_cat_ii {
        println!("Category II (medium comfort)");
    } else if en_result.acceptability_cat_iii {
        println!("Category III (acceptable comfort)");
    } else {
        println!("Outside acceptable range");
    }

    // Comparison
    println!("\n--- Interpretation ---");
    println!("Adaptive comfort models are designed for naturally ventilated buildings");
    println!("where occupants can adapt through clothing, activity, and window operation.");
    println!("\nThe running mean outdoor temperature ({:.1}°C) indicates that", t_running_mean.as_celsius());
    println!("occupants have adapted to cooler conditions, so a wider range of");
    println!("indoor temperatures is acceptable compared to mechanically cooled buildings.");
}
