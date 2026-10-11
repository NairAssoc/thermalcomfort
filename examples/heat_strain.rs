//! Heat and cold strain models example
//!
//! Demonstrates the rational (heat-balance) strain models: PHS (ISO 7933) for hot work,
//! IREQ (ISO 11079) for cold work, PET as an outdoor comfort temperature, and the sports
//! heat stress risk scale.

use std::time::Duration;
use thermalcomfort::models::{
    DurationLimitedExposure, IreqInputs, PetInputs, PhsInputs, PhsPosture, Sports,
    SportsHeatStressRiskInputs, ireq, pet_steady, phs, sports_heat_stress_risk,
};
use thermalcomfort::{
    AirPermeability, ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature,
};

fn main() {
    println!("=== Heat and Cold Strain Models ===\n");

    // --- PHS ---
    println!("--- PHS (Predicted Heat Strain, ISO 7933) ---");
    // Moderate work in a hot, dry workshop with a little air movement.
    let phs_tdb = Temperature::from_celsius(40.0);
    let phs_rh = Humidity::from_percent(33.85);
    println!(
        "Conditions: {:.0}°C air and radiant, {:.0}% RH, 0.3 m/s, 2.5 met, 0.5 clo, standing\n",
        phs_tdb.as_celsius(),
        phs_rh.as_percent()
    );
    let strain = phs(
        PhsInputs {
            tdb: phs_tdb,
            tr: phs_tdb,
            v: Speed::from_meters_per_second(0.3),
            rh: phs_rh,
            met: MetabolicRate::from_met(2.5),
            clo: ClothingInsulation::from_clo(0.5),
            posture: PhsPosture::Standing,
        },
        Default::default(),
    );
    println!("After the default 8 h shift:");
    println!(
        "  Rectal temperature:        {:.1}°C",
        strain.t_re.as_celsius()
    );
    println!(
        "  Skin temperature:          {:.1}°C",
        strain.t_sk.as_celsius()
    );
    println!(
        "  Sweat loss:                {:.0} g",
        strain.sweat_loss_g.as_grams()
    );
    // The limits are `None` only when the inputs were outside ISO 7933's range.
    println!("Exposure limits:");
    println!(
        "  Rectal temperature 38 °C:  {}",
        describe_limit(strain.d_lim_t_re)
    );
    println!(
        "  Water loss, 50% of people: {}",
        describe_limit(strain.d_lim_loss_50)
    );
    println!(
        "  Water loss, 95% of people: {}",
        describe_limit(strain.d_lim_loss_95)
    );

    // --- IREQ ---
    println!("\n--- IREQ (Required Clothing Insulation, ISO 11079) ---");
    // Outdoor work at -15 °C in a 2 m/s wind. The model reports the insulation needed
    // for thermal balance and, when the worker has less, how long they can stay out.
    let cold = Temperature::from_celsius(-15.0);
    for clo_worn in [2.8, 1.5] {
        let result = ireq(
            IreqInputs {
                tdb: cold,
                tr: cold,
                vr: Speed::from_meters_per_second(2.0),
                rh: Humidity::from_percent(55.0),
                met: MetabolicRate::from_w_per_m2(175.0),
                clo: ClothingInsulation::from_clo(clo_worn),
                p: AirPermeability::from_l_per_m2_s(50.0),
                walk_sp: Speed::from_meters_per_second(1.1),
            },
            Default::default(),
        );
        println!(
            "{:.0}°C, 175 W/m², wearing {clo_worn:.1} clo:",
            cold.as_celsius()
        );
        println!(
            "  Required insulation: {:.1} clo (minimal), {:.1} clo (neutral)",
            result.ireq_min.as_clo(),
            result.ireq_neutral.as_clo()
        );
        println!(
            "  Exposure limit (minimal): {}",
            describe_dle(result.dle_min)
        );
        println!(
            "  Exposure limit (neutral): {}",
            describe_dle(result.dle_neutral)
        );
    }

    // --- PET ---
    println!("\n--- PET (Physiological Equivalent Temperature) ---");
    // The air temperature of a reference indoor room in which the body reaches the
    // same core and skin temperatures as outdoors. Same person on two days.
    let scenarios = [
        ("Mild spring day", 25.0, 27.0, 1.0, 50.0, 1.0),
        ("Winter afternoon", 2.0, -2.0, 3.0, 70.0, 1.5),
    ];
    for (label, tdb, tr, v, rh, clo) in scenarios {
        let result = pet_steady(
            PetInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                v: Speed::from_meters_per_second(v),
                rh: Humidity::from_percent(rh),
                met: MetabolicRate::from_met(1.5),
                clo: ClothingInsulation::from_clo(clo),
            },
            Default::default(),
        );
        println!(
            "  {label}: {tdb:.0}°C air, {tr:.0}°C radiant, {v:.1} m/s, {clo:.1} clo -> PET {:.1}°C",
            result.pet
        );
    }

    // --- Sports heat stress risk ---
    println!("\n--- Sports heat stress risk ---");
    // Each sport carries its own clothing, metabolic rate and duration. The risk level
    // runs from 1 (low) to 4 (extreme), interpolated within each band.
    for tdb in [24.0, 35.0] {
        let result = sports_heat_stress_risk(SportsHeatStressRiskInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tdb),
            rh: Humidity::from_percent(40.0),
            vr: Speed::from_meters_per_second(0.1),
            sport: Sports::RUNNING,
        })
        .expect("a finite air temperature always falls in one of the risk bands");
        println!(
            "Running at {tdb:.0}°C, 40% RH: risk level {:.1}",
            result.risk_level_interpolated
        );
        println!(
            "  Band thresholds: medium {:.1}°C, high {:.1}°C, extreme {:.1}°C",
            result.t_medium.as_celsius(),
            result.t_high.as_celsius(),
            result.t_extreme.as_celsius()
        );
        println!("  {}", result.recommendation);
    }
}

/// Render a PHS exposure limit in minutes.
fn describe_limit(limit: Option<Duration>) -> String {
    match limit {
        Some(d) => format!("{:.0} min", d.as_secs_f64() / 60.0),
        None => "outside ISO 7933 limits".to_string(),
    }
}

/// Render a duration-limited exposure in words.
fn describe_dle(dle: DurationLimitedExposure) -> String {
    match dle {
        DurationLimitedExposure::Limited(d) => format!("{:.1} h", d.as_secs_f64() / 3600.0),
        DurationLimitedExposure::MoreThanEight => "more than 8 h".to_string(),
        DurationLimitedExposure::NotApplicable => "outside ISO 11079 limits".to_string(),
    }
}
