//! Solar gain and local discomfort example
//!
//! Demonstrates the ASHRAE 55 solar gain calculation (with the sky-vault view fraction
//! and the supine angle transposition it depends on) and the local discomfort models
//! for ankle draft and vertical temperature gradient.

use thermalcomfort::models::{
    AnkleDraftInputs, FSvvInputs, SolarGainInputs, SolarGainOptions, SolarGainPosture,
    VerticalTmpGradPpdInputs, ankle_draft, f_svv, solar_gain, transpose_sharp_altitude,
    vertical_tmp_grad_ppd,
};
use thermalcomfort::{
    Angle, ClothingInsulation, HeatFluxDensity, Humidity, Length, MetabolicRate, Speed,
    Temperature, TemperatureDelta,
};

fn main() {
    println!("=== Solar Gain and Local Discomfort ===\n");

    // --- Sky-vault view fraction ---
    println!("--- Sky-vault view fraction ---");
    // How much of the sky an occupant sees through a window: 2 m wide, 1.5 m tall,
    // seen from 3 m back in the room.
    let window = FSvvInputs {
        width: Length::from_meters(2.0),
        height: Length::from_meters(1.5),
        distance: Length::from_meters(3.0),
    };
    let f_svv_window = f_svv(window);
    println!(
        "  {:.1} m x {:.1} m window at {:.1} m: f_svv = {f_svv_window:.3}",
        window.width.as_meters(),
        window.height.as_meters(),
        window.distance.as_meters()
    );

    // --- Solar gain ---
    println!("\n--- Solar gain (ASHRAE 55 Appendix C) ---");
    // Mid-morning sun through that window: 45° above the horizon, 90° to the
    // occupant's side, 600 W/m² direct beam, 70% transmittance, 70% of the body lit.
    let sol_altitude = Angle::from_degrees(45.0);
    let sharp = Angle::from_degrees(90.0);
    let solar = SolarGainInputs {
        sol_altitude,
        sharp,
        sol_radiation_dir: HeatFluxDensity::from_watts_per_square_meter(600.0),
        sol_transmittance: 0.7,
        f_svv: f_svv_window,
        f_bes: 0.7,
    };
    println!(
        "Sun at {:.0}° altitude, {:.0}° from the front, {:.0} W/m² direct beam\n",
        sol_altitude.as_degrees(),
        sharp.as_degrees(),
        solar.sol_radiation_dir.as_watts_per_square_meter()
    );

    for posture in [
        SolarGainPosture::Sitting,
        SolarGainPosture::Standing,
        SolarGainPosture::Supine,
    ] {
        let gain = solar_gain(
            solar,
            SolarGainOptions {
                posture,
                ..Default::default()
            },
        );
        println!(
            "  {:<9} ERF {:>5.1} W/m², mean radiant temperature raised by {:.1} K",
            format!("{posture:?}"),
            gain.erf.as_watts_per_square_meter(),
            gain.delta_mrt.as_celsius()
        );
    }

    // The supine case reuses the standing projected-area table by rotating the sun's
    // position into the lying body's frame. This is that rotation on its own.
    let (sharp_t, altitude_t) = transpose_sharp_altitude(sharp, sol_altitude);
    println!(
        "\n  Transposed for a supine body: sharp {:.1}° -> {:.1}°, altitude {:.1}° -> {:.1}°",
        sharp.as_degrees(),
        sharp_t.as_degrees(),
        sol_altitude.as_degrees(),
        altitude_t.as_degrees()
    );

    // --- Ankle draft ---
    println!("\n--- Ankle draft (ASHRAE 55) ---");
    // Seated in a 23 °C office with still air at head height; the air speed at ankle
    // height decides the draft complaint rate. Acceptable is PPD < 20%.
    let office_tdb = Temperature::from_celsius(23.0);
    for v_ankle_ms in [0.1, 0.25, 0.4] {
        let (ppd, acceptable) = ankle_draft(
            AnkleDraftInputs {
                tdb: office_tdb,
                tr: office_tdb,
                vr: Speed::from_meters_per_second(0.1),
                rh: Humidity::from_percent(45.0),
                met: MetabolicRate::from_met(1.1),
                clo: ClothingInsulation::from_clo(0.7),
                v_ankle: Speed::from_meters_per_second(v_ankle_ms),
            },
            Default::default(),
        );
        println!("  {v_ankle_ms:.2} m/s at the ankles: PPD {ppd:.1}%, acceptable: {acceptable}");
    }

    // --- Vertical temperature gradient ---
    println!("\n--- Vertical air temperature gradient (ISO 7730 / ASHRAE 55) ---");
    // Warm head, cool feet: the difference between 1.1 m and 0.1 m above the floor.
    // Acceptable is PPD <= 5%. The ASHRAE 55 fit is an offset logistic curve, so for
    // small gradients it reports a slightly negative percentage; read that as "no one
    // dissatisfied".
    for gradient_k in [3.0, 5.0, 7.0] {
        let (ppd, acceptable) = vertical_tmp_grad_ppd(
            VerticalTmpGradPpdInputs {
                tdb: Temperature::from_celsius(25.0),
                tr: Temperature::from_celsius(25.0),
                vr: Speed::from_meters_per_second(0.1),
                rh: Humidity::from_percent(50.0),
                met: MetabolicRate::from_met(1.2),
                clo: ClothingInsulation::from_clo(0.5),
                vertical_tmp_grad: TemperatureDelta::from_celsius(gradient_k),
            },
            Default::default(),
        );
        println!(
            "  {gradient_k:.1} K between head and ankles: PPD {ppd:.1}%, acceptable: {acceptable}"
        );
    }
}
