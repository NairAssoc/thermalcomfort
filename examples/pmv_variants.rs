//! PMV model variants example
//!
//! Demonstrates the PMV family beyond the plain ISO 7730 call: the ASHRAE 55 variant
//! with its airspeed-control rule, the adaptive (pmv_a), expectancy (pmv_e) and ATHB
//! corrections for non-air-conditioned buildings, the cooling effect of elevated air
//! speed, and what the `limit_inputs` / `round_output` options change.

use thermalcomfort::models::{
    CoolingEffectInputs, PmvAInputs, PmvAthbInputs, PmvAthbOptions, PmvEInputs,
    PmvPpdAshraeOptions, PmvPpdInputs, PmvPpdIsoOptions, cooling_effect, pmv_a, pmv_athb, pmv_e,
    pmv_ppd_ashrae, pmv_ppd_iso,
};
use thermalcomfort::utilities::v_relative;
use thermalcomfort::{ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature};

fn main() {
    println!("=== PMV Model Variants ===\n");

    // A warm office: light summer clothing, seated work, a desk fan running.
    let tdb = Temperature::from_celsius(28.0);
    let tr = Temperature::from_celsius(28.0);
    let rh = Humidity::from_percent(50.0);
    let met = MetabolicRate::from_met(1.1);
    let clo = ClothingInsulation::from_clo(0.5);
    let v_still = Speed::from_meters_per_second(0.1);
    let v_fan = Speed::from_meters_per_second(0.9);

    println!(
        "Conditions: {:.0}°C air and radiant, {:.0}% RH, {:.1} met, {:.1} clo\n",
        tdb.as_celsius(),
        rh.as_percent(),
        met.as_met(),
        clo.as_clo()
    );

    // --- ASHRAE 55 with the fan on ---
    println!("--- ASHRAE 55: elevated air speed ---");
    let vr_fan = v_relative(v_fan, met);
    let inputs_fan = PmvPpdInputs {
        tdb,
        tr,
        vr: vr_fan,
        rh,
        met,
        clo,
    };

    // Occupant controls the fan (upstream's default): ASHRAE 55 allows the speed.
    let controlled = pmv_ppd_ashrae(inputs_fan, Default::default());
    println!(
        "  Occupant-controlled fan at {:.1} m/s: PMV {:.2}, PPD {:.1}%, compliant: {}",
        vr_fan.as_meters_per_second(),
        controlled.pmv,
        controlled.ppd,
        controlled
            .compliance
            .map_or("n/a".to_string(), |c| c.to_string())
    );

    // Nobody controls the fan: §7.2.1.2 forbids >0.8 m/s with clo < 0.7 and met < 1.3,
    // so the result is NaN rather than a number the standard does not cover.
    let uncontrolled = pmv_ppd_ashrae(
        inputs_fan,
        PmvPpdAshraeOptions {
            airspeed_control: false,
            ..Default::default()
        },
    );
    println!(
        "  Same fan, no occupant control:     PMV {}, compliant: {}",
        if uncontrolled.pmv.is_nan() {
            "NaN (outside ASHRAE 55 applicability)".to_string()
        } else {
            format!("{:.2}", uncontrolled.pmv)
        },
        uncontrolled
            .compliance
            .map_or("n/a".to_string(), |c| c.to_string())
    );

    // --- ISO 7730: limit_inputs and round_output ---
    println!("\n--- ISO 7730: input limits and rounding ---");
    let hot = Temperature::from_celsius(32.0); // above ISO 7730's 30 °C ceiling
    let vr_still = v_relative(v_still, met);
    let inputs_hot = PmvPpdInputs {
        tdb: hot,
        tr: hot,
        vr: vr_still,
        rh,
        met,
        clo,
    };

    let limited = pmv_ppd_iso(inputs_hot, Default::default());
    println!(
        "  {:.0}°C with limit_inputs=true:  PMV {}",
        hot.as_celsius(),
        if limited.pmv.is_nan() {
            "NaN (tdb outside 10-30 °C)".to_string()
        } else {
            format!("{:.2}", limited.pmv)
        }
    );

    let extrapolated = pmv_ppd_iso(
        inputs_hot,
        PmvPpdIsoOptions {
            limit_inputs: false,
            round_output: false,
            ..Default::default()
        },
    );
    println!(
        "  {:.0}°C with limit_inputs=false, round_output=false: PMV {:.4}, PPD {:.3}%",
        hot.as_celsius(),
        extrapolated.pmv,
        extrapolated.ppd
    );
    println!("  (the standard's equations still evaluate outside their validated range;");
    println!("   the flag only controls whether the crate reports that as NaN)");

    // --- Cooling effect ---
    println!("\n--- Cooling effect of elevated air speed ---");
    // The drop in operative temperature, at still air, that gives the same heat loss
    // as the fan does. ASHRAE 55 subtracts it from tdb and tr before computing PMV.
    let ce = cooling_effect(
        CoolingEffectInputs {
            tdb,
            tr,
            vr: vr_fan,
            rh,
            met,
            clo,
        },
        Default::default(),
    );
    println!(
        "  {:.1} m/s relative air speed cools like a {:.2} K drop in temperature",
        vr_fan.as_meters_per_second(),
        ce.as_celsius()
    );

    // --- Adaptive and expectancy corrections ---
    println!("\n--- Corrections for non-air-conditioned buildings ---");
    let warm = Temperature::from_celsius(30.0);
    let inputs_warm = PmvPpdInputs {
        tdb: warm,
        tr: warm,
        vr: vr_still,
        rh,
        met,
        clo,
    };
    let reference = pmv_ppd_iso(inputs_warm, Default::default());
    println!(
        "  Reference PMV at {:.0}°C, still air: {:.2}",
        warm.as_celsius(),
        reference.pmv
    );

    // pmv_a (Yao et al.): PMV scaled by an adaptive coefficient λ fitted to field
    // studies. 0.293 is the published value for warm, naturally ventilated buildings.
    let a_coefficient = 0.293;
    let adaptive = pmv_a(
        PmvAInputs {
            tdb: warm,
            tr: warm,
            vr: vr_still,
            rh,
            met,
            clo,
            a_coefficient,
        },
        Default::default(),
    );
    println!("  Adaptive PMV (λ = {a_coefficient}):     {adaptive:.2}");

    // pmv_e (Fanger & Toftum): PMV scaled by an expectancy factor, 0.5-0.7 for
    // regions where air conditioning is rare, with a metabolic rate reduced for
    // the warmth. 0.6 suits a warm climate with some air-conditioned buildings.
    let e_coefficient = 0.6;
    let expectancy = pmv_e(
        PmvEInputs {
            tdb: warm,
            tr: warm,
            vr: vr_still,
            rh,
            met,
            clo,
            e_coefficient,
        },
        Default::default(),
    );
    println!("  Expectancy PMV (e = {e_coefficient}):    {expectancy:.2}");

    // pmv_athb (Schweiker et al.): physiological, behavioural and psychological
    // adaptation driven by the running mean outdoor temperature. Clothing is derived
    // from that temperature by default, or taken as given.
    let t_running_mean = Temperature::from_celsius(26.0);
    let athb_derived = pmv_athb(
        PmvAthbInputs {
            tdb: warm,
            tr: warm,
            vr: vr_still,
            rh,
            met,
            t_running_mean,
        },
        Default::default(),
    );
    let athb_fixed = pmv_athb(
        PmvAthbInputs {
            tdb: warm,
            tr: warm,
            vr: vr_still,
            rh,
            met,
            t_running_mean,
        },
        PmvAthbOptions { clo: Some(clo) },
    );
    println!(
        "  ATHB PMV (outdoor running mean {:.0}°C):",
        t_running_mean.as_celsius()
    );
    println!("    clothing adapted to outdoor temperature: {athb_derived:.2}");
    println!(
        "    clothing fixed at {:.1} clo:               {athb_fixed:.2}",
        clo.as_clo()
    );
}
