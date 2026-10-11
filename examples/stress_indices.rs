//! Heat and cold stress indices example
//!
//! Demonstrates various thermal stress indices for extreme conditions.

use thermalcomfort::models::{
    AtInputs, AtOptions, DiscomfortIndexInputs, EsiInputs, HeatIndexLuInputs,
    HeatIndexRothfuszInputs, HeatIndexSchoenInputs, HumidexInputs, NetInputs, ThiInputs, WciInputs,
    WindChillTemperatureInputs, at, discomfort_index, esi, heat_index_lu, heat_index_rothfusz,
    heat_index_schoen, humidex, net, thi, wci, wind_chill_temperature,
};
use thermalcomfort::{HeatFluxDensity, Humidity, Speed, Temperature};

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
    let hi_result = heat_index_rothfusz(
        HeatIndexRothfuszInputs {
            tdb: hot_temp,
            rh: hot_rh,
        },
        Default::default(),
    );
    println!("Heat Index (Rothfusz):");
    println!("  {:.1}°C", hi_result.hi);
    if let Some(cat) = hi_result.stress_category {
        println!("  {}", cat.as_str());
    } else {
        println!("  (outside applicability range)");
    }

    // The category is judged on the unrounded index, so an index that rounds onto a
    // band edge still carries the band it is really in: 30.645 °C at 50 % RH gives a
    // heat index of 31.95 °C, reported as 32.0, in "extreme caution" rather than the
    // "caution" a reading of 32.0 would suggest.
    let on_the_edge = heat_index_rothfusz(
        HeatIndexRothfuszInputs {
            tdb: Temperature::from_celsius(30.645),
            rh: Humidity::from_percent(50.0),
        },
        Default::default(),
    );
    println!(
        "  At 30.645°C: {:.1}°C, {}",
        on_the_edge.hi,
        on_the_edge
            .stress_category
            .map_or("(uncategorised)", |c| c.as_str())
    );

    // Humidex
    let humidex_result = humidex(
        HumidexInputs {
            tdb: hot_temp,
            rh: hot_rh,
        },
        Default::default(),
    );
    println!("\nHumidex (Canadian):");
    println!("  {:.0}", humidex_result.humidex);
    println!("  {}", humidex_result.discomfort.as_str());

    // Temperature-Humidity Index (THI)
    let thi_val = thi(
        ThiInputs {
            tdb: hot_temp,
            rh: hot_rh,
        },
        Default::default(),
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
    let di_result = discomfort_index(DiscomfortIndexInputs {
        tdb: hot_temp,
        rh: hot_rh,
    });
    println!("\nDiscomfort Index:");
    println!("  {:.1}", di_result.di);
    println!("  {}", di_result.discomfort_condition.as_str());

    // Heat Index (Schoen): the same apparent-temperature idea as Rothfusz, fitted
    // through the dew point, with no dry-bulb applicability gate so the category is
    // always available.
    let schoen = heat_index_schoen(
        HeatIndexSchoenInputs {
            tdb: hot_temp,
            rh: hot_rh,
        },
        Default::default(),
    );
    println!("\nHeat Index (Schoen):");
    println!("  {:.1}°C", schoen.hi);
    println!(
        "  {}",
        schoen
            .stress_category
            .map_or("(uncategorised)", |c| c.as_str())
    );

    // Heat Index (Lu & Romps): a physically based extension of Steadman's model
    // that stays defined in the extreme heat where the empirical fits break down.
    // It reports a temperature only, with no stress category.
    let lu = heat_index_lu(
        HeatIndexLuInputs {
            tdb: hot_temp,
            rh: hot_rh,
        },
        Default::default(),
    );
    println!("\nHeat Index (Lu & Romps):");
    println!("  {:.1}°C", lu.hi);

    // Wind-dependent heat indices: a light breeze at 1.2 m (NET) / 10 m (AT)
    let hot_wind = Speed::from_meters_per_second(1.0);
    println!("\nWith {:.1} m/s wind:", hot_wind.as_meters_per_second());

    // Normal Effective Temperature (Missenard)
    let net_val = net(
        NetInputs {
            tdb: hot_temp,
            rh: hot_rh,
            v: hot_wind,
        },
        Default::default(),
    );
    let net_c = net_val.as_celsius();
    println!("  Normal Effective Temperature: {net_c:.1}°C");
    if net_c > 27.0 {
        println!("    Hot");
    } else if net_c > 23.0 {
        println!("    Warm");
    } else {
        println!("    Comfortable or cooler");
    }

    // Apparent Temperature (Steadman / Australian BoM), in shade and in sun.
    // `q` is the net radiation absorbed per unit body surface.
    let at_shade = at(
        AtInputs {
            tdb: hot_temp,
            rh: hot_rh,
            v: hot_wind,
        },
        Default::default(),
    );
    let at_sun = at(
        AtInputs {
            tdb: hot_temp,
            rh: hot_rh,
            v: hot_wind,
        },
        AtOptions {
            q: Some(HeatFluxDensity::from_watts_per_square_meter(200.0)),
            ..Default::default()
        },
    );
    println!(
        "  Apparent Temperature (shade):      {:.1}°C",
        at_shade.as_celsius()
    );
    println!(
        "  Apparent Temperature (200 W/m² sun): {:.1}°C",
        at_sun.as_celsius()
    );

    // Environmental Stress Index (Moran): temperature, humidity and global solar
    // radiation, designed to track WBGT without a globe thermometer.
    let sol_radiation_global = HeatFluxDensity::from_watts_per_square_meter(800.0); // clear midday sky
    let esi_val = esi(
        EsiInputs {
            tdb: hot_temp,
            rh: hot_rh,
            sol_radiation_global,
        },
        Default::default(),
    );
    println!(
        "  Environmental Stress Index ({:.0} W/m² solar): {esi_val:.1}",
        sol_radiation_global.as_watts_per_square_meter()
    );

    // COLD STRESS INDICES
    println!("\n\n--- Cold Stress Assessment ---\n");

    let cold_temp = Temperature::from_celsius(-10.0);
    let v = Speed::from_kilometers_per_hour(20.0);

    println!(
        "Conditions: {:.0}°C, {:.0} km/h wind\n",
        cold_temp.as_celsius(),
        v.as_kilometers_per_hour()
    );

    // Wind Chill Temperature
    let wct = wind_chill_temperature(
        WindChillTemperatureInputs { tdb: cold_temp, v },
        Default::default(),
    );
    let wct_c = wct.as_celsius();
    println!("Wind Chill Temperature:");
    println!("  {wct_c:.1}°C (feels like)");
    if wct_c > -10.0 {
        println!("  Low risk");
    } else if wct_c > -28.0 {
        println!("  Low risk of frostbite");
    } else if wct_c > -40.0 {
        println!("  Moderate risk: frostbite in 10-30 minutes");
    } else if wct_c > -48.0 {
        println!("  High risk: frostbite in 5-10 minutes");
    } else {
        println!("  Extreme risk: frostbite in 2-5 minutes");
    }

    // Wind Chill Index
    let wci_val =
        wci(WciInputs { tdb: cold_temp, v }, Default::default()).as_watts_per_square_meter();
    println!("\nWind Chill Index:");
    println!("  {wci_val:.0} W/m²");
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
        let hi = heat_index_rothfusz(
            HeatIndexRothfuszInputs { tdb: temp_30, rh },
            Default::default(),
        );
        let hum = humidex(HumidexInputs { tdb: temp_30, rh }, Default::default());

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
