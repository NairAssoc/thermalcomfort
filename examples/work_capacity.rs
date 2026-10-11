//! Work capacity and body temperature prediction example
//!
//! Demonstrates the four work-capacity curves (the share of an hour that can be
//! worked at a given WBGT) and the ridge-regression predictor of rectal and skin
//! temperature over a hot exposure.

use std::time::Duration;
use thermalcomfort::models::{
    RidgeRegressionOptions, WorkCapacityIntensityOptions, WorkIntensity,
    ridge_regression_predict_t_re_t_sk, work_capacity_dunne, work_capacity_hothaps,
    work_capacity_iso, work_capacity_niosh,
};
use thermalcomfort::{Age, Humidity, Length, Mass, Power, Sex, Temperature};

fn main() {
    println!("=== Work Capacity Under Heat Stress ===\n");

    // --- ISO and NIOSH ---
    println!("--- ISO 7243 and NIOSH work/rest curves ---");
    // Both relate WBGT to the metabolic rate the standard allows for continuous work.
    // 300 W is moderate manual labour.
    let metabolic_power = Power::from_watts(300.0);
    println!(
        "Work capacity [%] at {:.0} W metabolic rate:\n",
        metabolic_power.as_watts()
    );
    println!("  WBGT     ISO    NIOSH");
    for wbgt_c in [24.0, 26.0, 28.0, 30.0, 32.0, 34.0] {
        let wbgt = Temperature::from_celsius(wbgt_c);
        println!(
            "  {wbgt_c:.0}°C   {:>5.1}  {:>5.1}",
            work_capacity_iso(wbgt, metabolic_power),
            work_capacity_niosh(wbgt, metabolic_power)
        );
    }

    // --- Dunne and HOTHAPS ---
    println!("\n--- Dunne et al. and HOTHAPS (Kjellstrom et al.) ---");
    // These take a work-intensity class instead of a metabolic rate. HOTHAPS never
    // drops below 10%, on the assumption that some work is always possible in bursts.
    println!("Work capacity [%] by intensity:\n");
    println!("  WBGT    Dunne light/moderate/heavy    HOTHAPS light/moderate/heavy");
    for wbgt_c in [26.0, 30.0, 34.0, 38.0] {
        let wbgt = Temperature::from_celsius(wbgt_c);
        let light = WorkCapacityIntensityOptions {
            work_intensity: WorkIntensity::Light,
        };
        let moderate = WorkCapacityIntensityOptions {
            work_intensity: WorkIntensity::Moderate,
        };
        let heavy = WorkCapacityIntensityOptions {
            work_intensity: WorkIntensity::Heavy,
        };
        println!(
            "  {wbgt_c:.0}°C   {:>5.1} / {:>5.1} / {:>5.1}          {:>5.1} / {:>5.1} / {:>5.1}",
            work_capacity_dunne(wbgt, light),
            work_capacity_dunne(wbgt, moderate),
            work_capacity_dunne(wbgt, heavy),
            work_capacity_hothaps(wbgt, light),
            work_capacity_hothaps(wbgt, moderate),
            work_capacity_hothaps(wbgt, heavy)
        );
    }

    // --- Ridge regression body temperature prediction ---
    println!("\n--- Ridge regression: rectal and skin temperature ---");
    // A 60-year-old man resting for an hour at 35 °C and 60% RH. The model returns one
    // prediction per minute.
    let minutes: usize = 60;
    let (sex, age) = (Sex::Male, Age::from_years(60.0));
    let (height, weight) = (Length::from_meters(1.80), Mass::from_kilograms(75.0));
    let (tdb, rh) = (
        Temperature::from_celsius(35.0),
        Humidity::from_percent(60.0),
    );
    let duration = Duration::from_secs(minutes as u64 * 60);
    println!(
        "Male, {:.0} years, {:.2} m, {:.0} kg at {:.0}°C, {:.0}% RH for {minutes} min\n",
        age.as_years(),
        height.as_meters(),
        weight.as_kilograms(),
        tdb.as_celsius(),
        rh.as_percent()
    );

    let predicted = ridge_regression_predict_t_re_t_sk(
        sex,
        age,
        height,
        weight,
        tdb,
        rh,
        duration,
        Default::default(),
    );
    println!("  Default initial temperatures:");
    for minute in [0, 15, 30, 45, minutes - 1] {
        println!(
            "    minute {:>2}: rectal {:.2}°C, skin {:.2}°C",
            minute + 1,
            predicted.t_re[minute].as_celsius(),
            predicted.t_sk[minute].as_celsius()
        );
    }

    // Start from measured temperatures instead of the model's defaults.
    let warmed_up = ridge_regression_predict_t_re_t_sk(
        sex,
        age,
        height,
        weight,
        tdb,
        rh,
        duration,
        RidgeRegressionOptions {
            t_re: Some(Temperature::from_celsius(37.4)),
            t_sk: Some(Temperature::from_celsius(34.5)),
            ..Default::default()
        },
    );
    println!("  Starting from 37.4 °C rectal and 34.5 °C skin:");
    println!(
        "    minute {minutes}: rectal {:.2}°C, skin {:.2}°C",
        warmed_up.t_re[minutes - 1].as_celsius(),
        warmed_up.t_sk[minutes - 1].as_celsius()
    );
}
