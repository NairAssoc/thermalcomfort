//! Psychrometrics and body geometry example
//!
//! Demonstrates the moist-air property functions (saturation pressure in its several
//! forms, wet-bulb and dew-point temperature, humidity ratio and enthalpy) and body
//! surface area. The environmental temperatures derived from them (mean radiant and
//! operative) are in the `environment` example.

use thermalcomfort::psychrometrics::{
    PsyTaRhInputs, PsyTaRhOptions, antoine, dew_point_temperature, enthalpy_air, hr_to_rh, p_sat,
    p_sat_torr, psy_ta_rh, wet_bulb_temperature,
};
use thermalcomfort::utilities::{
    BodySurfaceAreaInputs, BodySurfaceAreaOptions, BsaFormula, body_surface_area,
    body_surface_area_dubois,
};
use thermalcomfort::{Humidity, Length, Mass, Pressure, Temperature};

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
    // Three formulations of the same quantity. p_sat is the ASHRAE Hyland-Wexler fit
    // used by the psychrometric routines; the Antoine and exponential (torr) forms are
    // the ones the PMV and two-node models use internally.
    println!("  p_sat (ASHRAE):     {:.1} Pa", p_sat(tdb).as_pascals());
    println!(
        "  antoine:            {:.3} kPa ({:.1} Pa)",
        antoine(tdb).as_kilopascals(),
        antoine(tdb).as_pascals()
    );
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
        "  hr_to_rh round trip:   {:.1}% (started from {:.0}%)",
        rh_back.as_percent(),
        rh.as_percent()
    );

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
