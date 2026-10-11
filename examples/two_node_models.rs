//! Two-node (Gagge) thermophysiological models example
//!
//! Demonstrates the Gagge two-node model and the indices built on it: SET (Standard
//! Effective Temperature), the Ji variant for older adults, the sleep variant driven by
//! per-minute schedules, and the heat-wave fan-use assessment.

use thermalcomfort::models::{
    GaggePosture, GaggeTwoNodesInputs, GaggeTwoNodesJiInputs, SetInputs, SetOptions, SleepInputs,
    UseFansHeatwavesInputs, set_tmp, two_nodes_gagge, two_nodes_gagge_ji, two_nodes_gagge_sleep,
    use_fans_heatwaves,
};
use thermalcomfort::utilities::p_sat_torr;
use thermalcomfort::{
    ClothingInsulation, Humidity, Length, MetabolicRate, Pressure, Speed, Temperature,
};

fn main() {
    println!("=== Two-Node Thermophysiological Models ===\n");

    // --- Two-node Gagge: a warm office ---
    println!("--- Two-node Gagge model ---");
    let tdb = Temperature::from_celsius(30.0);
    let tr = Temperature::from_celsius(30.0);
    let v = Speed::from_meters_per_second(0.3);
    let rh = Humidity::from_percent(50.0);
    let met = MetabolicRate::from_met(1.2);
    let clo = ClothingInsulation::from_clo(0.5);

    println!(
        "Conditions: {:.0}°C, {:.0}% RH, {:.1} m/s, {:.1} met, {:.1} clo\n",
        tdb.as_celsius(),
        rh.as_percent(),
        v.as_meters_per_second(),
        met.as_met(),
        clo.as_clo()
    );

    let gagge = two_nodes_gagge(
        GaggeTwoNodesInputs {
            tdb,
            tr,
            v,
            rh,
            met,
            clo,
        },
        Default::default(),
    );
    println!("Steady state after 60 minutes:");
    println!(
        "  Core temperature:       {:.2}°C",
        gagge.t_core.as_celsius()
    );
    println!(
        "  Skin temperature:       {:.2}°C",
        gagge.t_skin.as_celsius()
    );
    println!(
        "  Skin wettedness:        {:.2} (max {:.2})",
        gagge.w, gagge.w_max
    );
    println!(
        "  Evaporative heat loss:  {:.1} W/m²",
        gagge.e_skin.as_watts_per_square_meter()
    );
    println!(
        "  Sensible heat loss:     {:.1} W/m²",
        gagge.q_sensible.as_watts_per_square_meter()
    );
    println!("  SET:                    {:.1}°C", gagge.set.as_celsius());
    println!("  ET*:                    {:.1}°C", gagge.et.as_celsius());
    println!("  Thermal sensation:      {:.2}", gagge.t_sens);
    println!("  Discomfort:             {:.2}", gagge.disc);
    println!("  PMV (Gagge):            {:.2}", gagge.pmv_gagge);
    println!("  PMV (SET):              {:.2}", gagge.pmv_set);

    // --- SET ---
    println!("\n--- Standard Effective Temperature ---");
    // SET is the temperature of a standard environment (50% RH, still air, 1.0 clo
    // scaled to the activity) in which a person has the same skin temperature and
    // wettedness as in the actual one.
    let set_standing = set_tmp(
        SetInputs {
            tdb,
            tr,
            v,
            rh,
            met,
            clo,
        },
        Default::default(),
    );
    let set_sitting = set_tmp(
        SetInputs {
            tdb,
            tr,
            v,
            rh,
            met,
            clo,
        },
        SetOptions {
            position: GaggePosture::Sitting,
            ..Default::default()
        },
    );
    println!("  Standing: {:.1}°C", set_standing.as_celsius());
    println!("  Sitting:  {:.1}°C", set_sitting.as_celsius());

    // --- Ji variant for older adults ---
    println!("\n--- Two-node Gagge (Ji) for older adults ---");
    // The Ji model takes vapour pressure directly. Derive it from RH the way the
    // upstream documentation does: rh * p_sat_torr(tdb) / 100.
    let hot = Temperature::from_celsius(35.0);
    let hot_rh = Humidity::from_percent(40.0);
    let vapor_pressure =
        Pressure::from_torrs(hot_rh.as_percent() * p_sat_torr(hot).as_torrs() / 100.0);
    println!(
        "Conditions: {:.0}°C, {:.0}% RH ({:.1} torr vapour pressure), seated, 0.5 clo\n",
        hot.as_celsius(),
        hot_rh.as_percent(),
        vapor_pressure.as_torrs()
    );

    let ji = two_nodes_gagge_ji(
        GaggeTwoNodesJiInputs {
            tdb: hot,
            tr: hot,
            v: Speed::from_meters_per_second(0.1),
            met: MetabolicRate::from_met(1.0),
            clo,
            vapor_pressure,
        },
        Default::default(),
    );
    println!("Trajectory over the default 120 minute exposure:");
    for minute in [0, 30, 60, 90, 119] {
        println!(
            "  minute {:>3}: core {:.2}°C, skin {:.2}°C",
            minute + 1,
            ji.t_core[minute].as_celsius(),
            ji.t_skin[minute].as_celsius()
        );
    }

    // --- Sleep variant ---
    println!("\n--- Two-node Gagge (sleep) ---");
    // An eight-hour night, one entry per minute. The bedroom starts at 26 °C and is
    // cooled to 22 °C after four hours, at which point the sleeper pulls up a thicker
    // quilt. Every schedule must have the same length.
    let minutes = 8 * 60;
    let switch = 4 * 60;
    let tdb_night: Vec<Temperature> = (0..minutes)
        .map(|m| Temperature::from_celsius(if m < switch { 26.0 } else { 22.0 }))
        .collect();
    let tr_night = tdb_night.clone();
    let v_night = vec![Speed::from_meters_per_second(0.1); minutes];
    let rh_night = vec![Humidity::from_percent(50.0); minutes];
    let clo_night = vec![ClothingInsulation::from_clo(0.3); minutes]; // light pyjamas
    let quilt: Vec<Length> = (0..minutes)
        .map(|m| Length::from_centimeters(if m < switch { 2.0 } else { 5.0 }))
        .collect();

    let sleep = two_nodes_gagge_sleep(
        SleepInputs {
            tdb: &tdb_night,
            tr: &tr_night,
            v: &v_night,
            rh: &rh_night,
            clo: &clo_night,
            thickness_quilt: &quilt,
        },
        Default::default(),
    )
    .expect("all six per-minute schedules are the same length");

    println!("Hourly state (SET, core, skin, sensation, discomfort):");
    for hour in 1..=8 {
        let i = hour * 60 - 1;
        println!(
            "  {hour}h: SET {:.1}°C, core {:.2}°C, skin {:.2}°C, t_sens {:+.2}, disc {:.2}",
            sleep.set[i].as_celsius(),
            sleep.t_core[i].as_celsius(),
            sleep.t_skin[i].as_celsius(),
            sleep.t_sens[i],
            sleep.disc[i]
        );
    }

    // --- Fans during heat waves ---
    println!("\n--- Use of fans during heat waves ---");
    // Whether an electric fan still helps, or whether sweating and skin blood flow are
    // already at their limits so the extra convection only adds heat. In dry heat the
    // fan relieves the strain; in humid heat it cannot, because evaporation is capped.
    let heatwave_tdb = Temperature::from_celsius(45.0);
    for rh_percent in [30.0, 50.0] {
        let heatwave_rh = Humidity::from_percent(rh_percent);
        println!(
            "{:.0}°C, {:.0}% RH, seated at rest, 0.3 clo:",
            heatwave_tdb.as_celsius(),
            heatwave_rh.as_percent()
        );
        for v_ms in [0.2, 2.0, 4.0] {
            let fans = use_fans_heatwaves(
                UseFansHeatwavesInputs {
                    tdb: heatwave_tdb,
                    tr: heatwave_tdb,
                    v: Speed::from_meters_per_second(v_ms),
                    rh: heatwave_rh,
                    met: MetabolicRate::from_met(1.0),
                    clo: ClothingInsulation::from_clo(0.3),
                },
                Default::default(),
            );
            println!(
                "  {v_ms:.1} m/s: core {:.2}°C, skin {:.2}°C, sweat {:.0} g/h/m², wettedness {:.2}",
                fans.t_core.as_celsius(),
                fans.t_skin.as_celsius(),
                fans.m_rsw,
                fans.w
            );
            println!(
                "          heat strain: {} (blood flow {}, wettedness {}, sweating {})",
                flag(fans.heat_strain),
                flag(fans.heat_strain_blood_flow),
                flag(fans.heat_strain_w),
                flag(fans.heat_strain_sweating)
            );
        }
    }
}

/// Render an optional strain flag; `None` means the inputs were outside the model's range.
fn flag(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "n/a",
    }
}
