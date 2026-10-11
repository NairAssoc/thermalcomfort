//! Environment example
//!
//! Demonstrates the `environment` module: the quantities that describe the surroundings
//! before any comfort model runs. A weather-station wind is brought down to pedestrian
//! height with the logarithmic profile, a week of daily means becomes the running mean
//! outdoor temperature the adaptive models take, a globe reading becomes the mean
//! radiant temperature, air and radiant temperature combine into the operative
//! temperature, the air speed is corrected for the occupant's own movement, and a
//! window's sky-vault view and the sun's position are transposed for solar gain.

use thermalcomfort::environment::{
    FSvvInputs, MeanRadiantTemperatureInputs, MeanRadiantTemperatureOptions,
    OperativeTemperatureInputs, OperativeTemperatureOptions, RunningMeanOutdoorTemperatureOptions,
    ScaleWindSpeedLogError, ScaleWindSpeedLogOptions, f_svv, mean_radiant_temperature,
    operative_temperature, running_mean_outdoor_temperature, scale_wind_speed_log,
    transpose_sharp_altitude, v_relative,
};
use thermalcomfort::{Angle, Length, MetabolicRate, Speed, Temperature};

fn main() {
    println!("=== Environment ===\n");

    // --- Wind at the height of a person ---
    println!("--- Wind speed at pedestrian height ---");
    // A station anemometer at 10 m reads 4 m/s. The logarithmic profile gives the speed
    // at 1.1 m, the height ASHRAE 55 uses for a seated person, for terrains of
    // increasing roughness: the rougher the surface, the more the wind has died away by
    // the time it reaches the ground.
    let station = Speed::from_meters_per_second(4.0);
    let pedestrian = Length::from_meters(1.1);
    println!(
        "{:.1} m/s at 10 m, brought down to {:.1} m\n",
        station.as_meters_per_second(),
        pedestrian.as_meters()
    );
    for (terrain, z0_m) in [
        ("open water", 0.005),
        ("pasture", 0.03),
        ("cropland", 0.1),
        ("town", 0.4),
    ] {
        let v = scale_wind_speed_log(
            station,
            pedestrian,
            ScaleWindSpeedLogOptions {
                z0: Length::from_meters(z0_m),
                ..Default::default()
            },
        )
        .expect("1.1 m clears every roughness length in the table");
        println!(
            "  {terrain:<11} (z0 = {z0_m:>5} m): {:.2} m/s",
            v.as_meters_per_second()
        );
    }

    // In an urban area the roughness length is 1 m and the wind effectively dies at a
    // displacement height of 0.5 m among the buildings. The profile is undefined where
    // the target height, less the displacement, does not clear the roughness length,
    // and the function says which constraint failed rather than returning a number.
    let urban = scale_wind_speed_log(
        station,
        pedestrian,
        ScaleWindSpeedLogOptions {
            z0: Length::from_meters(1.0),
            d: Length::from_meters(0.5),
            ..Default::default()
        },
    );
    match urban {
        Ok(v) => println!("  urban: {:.2} m/s", v.as_meters_per_second()),
        Err(ScaleWindSpeedLogError::TargetMinusDisplacementNotAboveRoughness) => println!(
            "  urban (z0 = 1 m, d = 0.5 m): not defined at 1.1 m, the profile needs z2 - d > z0 and 0.6 m is not above 1 m"
        ),
        Err(e) => println!("  urban: not defined ({e})"),
    }

    // --- Running mean outdoor temperature ---
    println!("\n--- Running mean outdoor temperature ---");
    // Daily mean outdoor temperatures, most recent first, as the adaptive models want
    // them. EN 16798-1 weights yesterday at 0.8, the day before at 0.8², and so on.
    let daily_means: Vec<Temperature> = [21.5, 19.0, 17.5, 18.0, 22.0, 24.5, 23.0]
        .map(Temperature::from_celsius)
        .to_vec();
    println!(
        "Daily means, most recent first: {:?} °C\n",
        daily_means
            .iter()
            .map(|t| t.as_celsius())
            .collect::<Vec<_>>()
    );
    for (label, alpha) in [
        ("EN 16798-1 (α = 0.8)", 0.8),
        ("fast response (α = 0.6)", 0.6),
    ] {
        let t_rm = running_mean_outdoor_temperature(
            &daily_means,
            RunningMeanOutdoorTemperatureOptions {
                alpha,
                ..Default::default()
            },
        );
        println!("  {label:<24} {:.1}°C", t_rm.as_celsius());
    }

    // --- Mean radiant temperature from a globe thermometer ---
    println!("\n--- Mean radiant temperature from a globe reading ---");
    // A 150 mm black globe reads 30 °C in 26 °C air moving at 0.5 m/s. The default
    // method is the mixed-convection correlation; ISO 7726's forced/natural split is
    // available as an option.
    let globe = MeanRadiantTemperatureInputs {
        tg: Temperature::from_celsius(30.0),
        tdb: Temperature::from_celsius(26.0),
        v: Speed::from_meters_per_second(0.5),
    };
    println!(
        "Globe {:.0}°C, air {:.0}°C, {:.1} m/s, 150 mm globe\n",
        globe.tg.as_celsius(),
        globe.tdb.as_celsius(),
        globe.v.as_meters_per_second()
    );
    let tr_mixed = mean_radiant_temperature(globe, Default::default());
    let tr_iso = mean_radiant_temperature(
        globe,
        MeanRadiantTemperatureOptions {
            use_iso: true,
            ..Default::default()
        },
    );
    println!("  Mixed convection: {:.1}°C", tr_mixed.as_celsius());
    println!("  ISO 7726:         {:.1}°C", tr_iso.as_celsius());

    // --- Operative temperature ---
    println!("\n--- Operative temperature ---");
    // Air and radiant temperature weighted by air speed. The ISO form weights by
    // sqrt(10 v); ASHRAE 55 uses stepped coefficients.
    for v_ms in [0.1, 0.4, 1.0] {
        let inputs = OperativeTemperatureInputs {
            tdb: Temperature::from_celsius(24.0),
            tr: tr_mixed,
            v: Speed::from_meters_per_second(v_ms),
        };
        let to_iso = operative_temperature(inputs, Default::default());
        let to_ashrae =
            operative_temperature(inputs, OperativeTemperatureOptions { use_ashrae: true });
        println!(
            "  {v_ms:.1} m/s: ISO {:.1}°C, ASHRAE {:.1}°C",
            to_iso.as_celsius(),
            to_ashrae.as_celsius()
        );
    }

    // --- Relative air speed ---
    println!("\n--- Relative air speed ---");
    // A moving occupant feels more air than the anemometer measures: above 1 met the
    // speed is raised by 0.3 m/s per met of activity.
    let v_room = Speed::from_meters_per_second(0.1);
    for (activity, met) in [
        ("seated", 1.0),
        ("standing, light work", 1.6),
        ("walking", 2.0),
    ] {
        let vr = v_relative(v_room, MetabolicRate::from_met(met));
        println!(
            "  {activity:<20} {met:.1} met: {:.2} m/s",
            vr.as_meters_per_second()
        );
    }

    // --- Sky-vault view fraction and sun position ---
    println!("\n--- Window view and sun position ---");
    // The fraction of the sky an occupant sees through a 2 m × 1.5 m window from 1 m
    // away and from 3 m away, and the sun's position re-expressed for a person facing
    // the window: both feed the solar gain model.
    for distance_m in [1.0, 3.0] {
        let svv = f_svv(FSvvInputs {
            width: Length::from_meters(2.0),
            height: Length::from_meters(1.5),
            distance: Length::from_meters(distance_m),
        });
        println!("  2 m × 1.5 m window at {distance_m:.0} m: f_svv = {svv:.3}");
    }
    let (sharp, altitude) = (Angle::from_degrees(120.0), Angle::from_degrees(45.0));
    let (sharp_t, altitude_t) = transpose_sharp_altitude(sharp, altitude);
    println!(
        "  sun at SHARP {:.0}°, altitude {:.0}° → transposed SHARP {:.1}°, altitude {:.1}°",
        sharp.as_degrees(),
        altitude.as_degrees(),
        sharp_t.as_degrees(),
        altitude_t.as_degrees()
    );
}
