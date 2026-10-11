//! Psychrometrics and body geometry example
//!
//! Demonstrates the moist-air property functions (saturation pressure in its several
//! forms, wet-bulb and dew-point temperature, humidity ratio and enthalpy), the
//! derived environmental temperatures (mean radiant and operative), and body surface
//! area.

use thermalcomfort::psychrometrics::{
    MeanRadiantTemperatureInputs, MeanRadiantTemperatureOptions, OperativeTemperatureInputs,
    OperativeTemperatureOptions, PsyTaRhInputs, PsyTaRhOptions, dew_point_temperature,
    enthalpy_air, mean_radiant_temperature, operative_temperature, psy_ta_rh, wet_bulb_temperature,
};
use thermalcomfort::utilities::{
    BodySurfaceAreaInputs, BodySurfaceAreaOptions, BsaFormula, antoine, body_surface_area,
    body_surface_area_dubois, hr_to_rh, p_sat, p_sat_antoine, p_sat_torr,
};
use thermalcomfort::{Humidity, Length, Mass, Pressure, Speed, Temperature};

fn main() {
    println!("=== Psychrometrics ===\n");

    let tdb = Temperature::from_celsius(25.0);
    let rh = Humidity::from_percent(50.0);
    println!(
        "Moist air at {:.0}°C, {:.0}% RH\n",
        tdb.as_celsius(),
        rh.as_percent()
    );

    // --- Saturation vapour pressure ---
    println!("--- Saturation vapour pressure ---");
    // Four formulations of the same quantity. p_sat is the ASHRAE Hyland-Wexler fit
    // used by the psychrometric routines; the Antoine and exponential (torr) forms are
    // the ones the PMV and two-node models use internally.
    println!("  p_sat (ASHRAE):     {:.1} Pa", p_sat(tdb).as_pascals());
    println!(
        "  p_sat_antoine:      {:.1} Pa",
        p_sat_antoine(tdb).as_pascals()
    );
    println!("  antoine:            {:.3} kPa", antoine(tdb));
    println!(
        "  p_sat_torr:         {:.2} torr ({:.1} Pa)",
        p_sat_torr(tdb).as_torrs(),
        p_sat_torr(tdb).as_pascals()
    );

    // --- Full psychrometric state ---
    println!("\n--- Psychrometric state (psy_ta_rh) ---");
    let state = psy_ta_rh(PsyTaRhInputs { tdb, rh }, Default::default());
    println!("  Vapour pressure:      {:.1} Pa", state.p_vap.as_pascals());
    println!("  Humidity ratio:       {:.4} kg/kg dry air", state.hr);
    println!("  Wet-bulb temperature: {:.1}°C", state.t_wb.as_celsius());
    println!("  Dew point:            {:.1}°C", state.t_dp.as_celsius());
    println!(
        "  Enthalpy:             {:.1} kJ/kg dry air",
        state.h / 1000.0
    );

    // The same air at 1500 m altitude: a lower atmospheric pressure raises the
    // humidity ratio for the same relative humidity.
    let p_altitude = Pressure::from_pascals(84_556.0);
    let state_altitude = psy_ta_rh(
        PsyTaRhInputs { tdb, rh },
        PsyTaRhOptions { p_atm: p_altitude },
    );
    println!(
        "  At {:.1} kPa (1500 m): humidity ratio {:.4} kg/kg",
        p_altitude.as_pascals() / 1000.0,
        state_altitude.hr
    );

    // --- Individual properties ---
    println!("\n--- Individual properties ---");
    let t_wb = wet_bulb_temperature(tdb, rh);
    let t_dp = dew_point_temperature(tdb, rh);
    println!("  wet_bulb_temperature:  {:.1}°C", t_wb.as_celsius());
    println!("  dew_point_temperature: {:.1}°C", t_dp.as_celsius());

    // enthalpy_air takes the humidity ratio; hr_to_rh goes back the other way.
    let h = enthalpy_air(tdb, state.hr);
    let rh_back = hr_to_rh(state.hr, tdb, Pressure::from_pascals(101_325.0));
    println!(
        "  enthalpy_air at hr {:.4}: {:.1} kJ/kg",
        state.hr,
        h / 1000.0
    );
    println!(
        "  hr_to_rh round trip:   {rh_back:.1}% (started from {:.0}%)",
        rh.as_percent()
    );

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

    // --- Body surface area ---
    println!("\n--- Body surface area ---");
    let weight = Mass::from_kilograms(70.0);
    let height = Length::from_meters(1.75);
    println!(
        "{:.0} kg, {:.2} m\n",
        weight.as_kilograms(),
        height.as_meters()
    );
    println!(
        "  DuBois (direct):  {:.3} m²",
        body_surface_area_dubois(weight, height).as_square_meters()
    );
    for formula in [
        BsaFormula::DuBois,
        BsaFormula::Takahira,
        BsaFormula::Fujimoto,
        BsaFormula::Kurazumi,
    ] {
        let bsa = body_surface_area(
            BodySurfaceAreaInputs { weight, height },
            BodySurfaceAreaOptions { formula },
        );
        println!(
            "  {:<9}         {:.3} m²",
            format!("{formula:?}"),
            bsa.as_square_meters()
        );
    }
}
