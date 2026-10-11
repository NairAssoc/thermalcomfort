//! JOS3 multi-segment thermoregulation model example
//!
//! Demonstrates building a JOS3 subject with `Jos3Builder`, running a simulation in
//! phases whose environment changes between them, and reading both whole-body and
//! per-segment results.

use std::time::Duration;
use thermalcomfort::models::jos3::{Jos3Builder, Jos3Posture, PerBodyPart};
use thermalcomfort::{
    ActivityRatio, Age, BodyFat, ClothingInsulation, Humidity, Length, Mass, Sex, Speed,
    Temperature,
};

fn main() {
    println!("=== JOS3 Thermoregulation Model ===\n");

    // --- Subject ---
    println!("--- Subject ---");
    let mut sim = Jos3Builder::new()
        .height(Length::from_meters(1.75))
        .weight(Mass::from_kilograms(70.0))
        .age(Age::from_years(30.0))
        .fat(BodyFat::new(15.0).expect("15% body fat is within the 1-90% range"))
        .sex(Sex::Male)
        .build()
        .expect("height, weight, age and body fat are all within JOS3's validated ranges");

    println!(
        "  {:.2} m, {:.1} kg, age {:.0}, {:.0}% fat, {:?}",
        sim.height().as_meters(),
        sim.weight().as_kilograms(),
        sim.age().as_years(),
        sim.fat().as_percent(),
        sim.sex()
    );
    println!(
        "  Basal metabolic rate: {:.0} W/m²",
        sim.bmr().as_w_per_m2()
    );
    println!(
        "  Body surface area:    {:.2} m²",
        sim.bsa().iter().map(|a| a.as_square_meters()).sum::<f64>()
    );

    // --- Phase 1: an hour of seated office work ---
    println!("\n--- Phase 1: 60 min seated in a 24 °C office ---");
    // Start from the model's current environment and override only what changes.
    let mut conditions = sim.conditions();
    conditions.tdb = PerBodyPart::Uniform(Temperature::from_celsius(24.0));
    conditions.tr = PerBodyPart::Uniform(Temperature::from_celsius(24.0));
    conditions.rh = PerBodyPart::Uniform(Humidity::from_percent(50.0));
    conditions.v = PerBodyPart::Uniform(Speed::from_meters_per_second(0.1));
    conditions.clo = PerBodyPart::Uniform(ClothingInsulation::from_clo(0.6));
    conditions.par = ActivityRatio::from_ratio(1.2);
    conditions.posture = Jos3Posture::Sitting;
    sim.advance(&conditions, 60, Duration::from_secs(60))
        .expect("uniform conditions always resolve and par >= 1");
    report(&sim);

    // --- Phase 2: a brisk walk outdoors in the sun ---
    println!("\n--- Phase 2: 30 min walking outdoors, 32 °C air, 40 °C radiant ---");
    conditions.tdb = PerBodyPart::Uniform(Temperature::from_celsius(32.0));
    conditions.tr = PerBodyPart::Uniform(Temperature::from_celsius(40.0));
    conditions.rh = PerBodyPart::Uniform(Humidity::from_percent(45.0));
    conditions.v = PerBodyPart::Uniform(Speed::from_meters_per_second(1.0));
    conditions.par = ActivityRatio::from_ratio(2.5);
    conditions.posture = Jos3Posture::Standing;
    // Per-segment clothing: a hat and shirt, bare arms, shorts, shoes. Values are in
    // the order of `body_names()`.
    conditions.clo = PerBodyPart::BySegment(
        [
            0.1, // head
            0.0, // neck
            0.3, // chest
            0.3, // back
            0.5, // pelvis
            0.2, 0.0, 0.0, // left shoulder, arm, hand
            0.2, 0.0, 0.0, // right shoulder, arm, hand
            0.3, 0.0, 0.5, // left thigh, leg, foot
            0.3, 0.0, 0.5, // right thigh, leg, foot
        ]
        .map(ClothingInsulation::from_clo),
    );
    sim.advance(&conditions, 30, Duration::from_secs(60))
        .expect("per-segment conditions with 17 entries always resolve");
    report(&sim);

    // --- Phase 3: lying down in a cool room ---
    println!("\n--- Phase 3: 30 min lying in a 20 °C room ---");
    // `to` sets the operative temperature: it overrides both tdb and tr.
    conditions.to = Some(PerBodyPart::Uniform(Temperature::from_celsius(20.0)));
    conditions.v = PerBodyPart::Uniform(Speed::from_meters_per_second(0.1));
    conditions.rh = PerBodyPart::Uniform(Humidity::from_percent(50.0));
    conditions.par = ActivityRatio::from_ratio(1.0);
    conditions.posture = Jos3Posture::Lying;
    conditions.clo = PerBodyPart::Uniform(ClothingInsulation::from_clo(0.3));
    sim.advance(&conditions, 30, Duration::from_secs(60))
        .expect("operative-temperature override resolves like any uniform value");
    report(&sim);

    // --- History ---
    println!("\n--- Whole-body history ---");
    // results() holds one row per step plus the initial steady state.
    let results = sim.results();
    println!(
        "  {} rows ({} minutes simulated)",
        results.t_skin_mean.len(),
        results
            .simulation_time
            .last()
            .map_or(0.0, |s| s.as_secs_f64() / 60.0)
    );
    println!("  time   t_skin_mean  w_mean  cardiac output  heat production");
    for &minute in &[0usize, 30, 60, 75, 90, 105, 120] {
        println!(
            "  {:>3} min   {:.2}°C     {:.2}    {:.0} L/h       {:.0} W",
            minute,
            results.t_skin_mean[minute].as_celsius(),
            results.w_mean[minute],
            results.cardiac_output[minute],
            results.q_thermogenesis_total[minute].as_watts()
        );
    }
}

/// Print the current whole-body and per-segment state of the simulation.
fn report(sim: &thermalcomfort::models::jos3::Jos3Model) {
    println!(
        "  Mean skin temperature:   {:.2}°C",
        sim.t_skin_mean().as_celsius()
    );
    println!(
        "  Central blood:           {:.2}°C",
        sim.t_cb().as_celsius()
    );
    println!("  Core / skin by segment:");
    let names = sim.body_names();
    let t_core = sim.t_core();
    let t_skin = sim.t_skin();
    for (i, name) in names.iter().enumerate() {
        println!(
            "    {name:<15} core {:.2}°C   skin {:.2}°C",
            t_core[i].as_celsius(),
            t_skin[i].as_celsius()
        );
    }
}
