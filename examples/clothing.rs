//! Clothing insulation example
//!
//! Demonstrates building a clothing ensemble from garment tables, the area factor, and
//! the dynamic corrections for body movement and air speed from ASHRAE 55 and ISO 9920.

use thermalcomfort::clothing::{
    CloCorrectionFactorEnvironmentInputs, CloDynamicIsoOptions, CloInsulationAirLayerInputs,
    CloTotalInsulationInputs, clo_area_factor, clo_correction_factor_environment,
    clo_dynamic_ashrae, clo_dynamic_iso, clo_insulation_air_layer,
    clo_intrinsic_insulation_ensemble, clo_total_insulation,
};
use thermalcomfort::utilities::{clo_individual_garment, clo_tout, clo_typical_ensemble};
use thermalcomfort::{ClothingInsulation, MetabolicRate, Speed, Temperature};

fn main() {
    println!("=== Clothing Insulation ===\n");

    // --- Garment tables ---
    println!("--- Building an ensemble from individual garments ---");
    let garments = [
        "Men's underwear",
        "T-shirt",
        "Long-sleeve dress shirt",
        "Thin trousers",
        "Calf length socks",
        "Shoes or sandals",
    ];
    let mut garment_clos = Vec::new();
    for name in garments {
        match clo_individual_garment(name) {
            Some(clo) => {
                println!("  {name:<26} {:.2} clo", clo.as_clo());
                garment_clos.push(clo);
            }
            None => println!("  {name:<26} (not in the ASHRAE 55 garment table)"),
        }
    }
    // The ensemble is not the plain sum: garments overlap, so ISO 9920 fits
    // I_cl = 0.835 * sum + 0.161.
    let ensemble = clo_intrinsic_insulation_ensemble(&garment_clos);
    let plain_sum: f64 = garment_clos.iter().map(|c| c.as_clo()).sum();
    println!("  Plain sum:                 {plain_sum:.2} clo");
    println!("  Intrinsic ensemble:        {:.2} clo", ensemble.as_clo());

    println!("\n--- Typical ensembles ---");
    for name in [
        "Typical summer indoor clothing",
        "Trousers, long-sleeve shirt",
        "Typical winter indoor clothing",
        "Space suit",
    ] {
        match clo_typical_ensemble(name) {
            Some(clo) => println!("  {name:<32} {:.2} clo", clo.as_clo()),
            None => println!("  {name:<32} (not in the ensemble table)"),
        }
    }

    // --- Area factor ---
    println!("\n--- Clothing area factor ---");
    // Clothing enlarges the surface exchanging heat with the room: f_cl = 1 + 0.28 I_cl.
    for clo in [0.0, 0.5, 1.0, 1.5] {
        println!(
            "  {clo:.1} clo: f_cl = {:.2}",
            clo_area_factor(ClothingInsulation::from_clo(clo))
        );
    }

    // --- Dynamic insulation ---
    println!("\n--- Dynamic insulation: ASHRAE 55 ---");
    // Movement pumps air through clothing. ASHRAE 55 reduces the insulation above
    // 1.2 met and leaves it alone below.
    let clo_static = ClothingInsulation::from_clo(1.0);
    for met in [1.0, 1.2, 2.0, 3.0] {
        let dynamic =
            clo_dynamic_ashrae(clo_static, MetabolicRate::from_met(met), Default::default());
        println!(
            "  {:.1} clo at {met:.1} met -> {:.2} clo",
            clo_static.as_clo(),
            dynamic.as_clo()
        );
    }

    println!("\n--- Dynamic insulation: ISO 9920 ---");
    // ISO 9920 also accounts for air speed, and derives a walking speed from the
    // metabolic rate. The default boundary air layer is 0.7 clo.
    let met_walking = MetabolicRate::from_met(2.0);
    for v_ms in [0.1, 0.5, 1.5] {
        let dynamic = clo_dynamic_iso(
            clo_static,
            met_walking,
            Speed::from_meters_per_second(v_ms),
            Default::default(),
        );
        println!(
            "  {:.1} clo at {:.1} met in {v_ms:.1} m/s -> {:.2} clo",
            clo_static.as_clo(),
            met_walking.as_met(),
            dynamic.as_clo()
        );
    }
    let dynamic_thin_layer = clo_dynamic_iso(
        clo_static,
        met_walking,
        Speed::from_meters_per_second(0.5),
        CloDynamicIsoOptions {
            i_a: ClothingInsulation::from_clo(0.6),
            ..Default::default()
        },
    );
    println!(
        "  Same at 0.5 m/s with a 0.6 clo static air layer -> {:.2} clo",
        dynamic_thin_layer.as_clo()
    );

    // --- The ISO 9920 building blocks ---
    println!("\n--- ISO 9920 correction components ---");
    // Walking at 0.5 m/s (1.8 km/h) in a 0.5 m/s breeze, 0.7 clo static air layer.
    let vr = Speed::from_meters_per_second(0.5);
    let v_walk = Speed::from_meters_per_second(0.5);
    let i_a_static = ClothingInsulation::from_clo(0.7);
    println!(
        "Relative air speed {:.1} m/s, walking at {:.1} m/s\n",
        vr.as_meters_per_second(),
        v_walk.as_meters_per_second()
    );

    let i_a_dynamic = clo_insulation_air_layer(CloInsulationAirLayerInputs {
        vr,
        v_walk,
        i_a_static,
    });
    println!(
        "  Boundary air layer: {:.2} clo static -> {:.2} clo dynamic",
        i_a_static.as_clo(),
        i_a_dynamic.as_clo()
    );

    // The correction factor depends on how much clothing is worn: nude skin loses
    // more insulation to wind than a full ensemble, and 0 < I_cl <= 0.6 blends the two.
    for i_cl in [0.0, 0.3, 1.0] {
        let factor = clo_correction_factor_environment(CloCorrectionFactorEnvironmentInputs {
            vr,
            v_walk,
            i_cl: ClothingInsulation::from_clo(i_cl),
        });
        println!("  Correction factor for {i_cl:.1} clo: {factor:.3}");
    }

    // Total insulation (clothing plus air layer) under the same movement.
    let i_cl = clo_static;
    let i_t =
        ClothingInsulation::from_clo(i_cl.as_clo() + i_a_static.as_clo() / clo_area_factor(i_cl));
    let i_t_dynamic = clo_total_insulation(CloTotalInsulationInputs {
        i_t,
        vr,
        v_walk,
        i_a_static,
        i_cl,
    });
    println!(
        "  Total insulation: {:.2} clo static -> {:.2} clo dynamic",
        i_t.as_clo(),
        i_t_dynamic.as_clo()
    );

    // --- Clothing from outdoor temperature ---
    println!("\n--- Clothing people choose for the outdoor temperature (clo_tout) ---");
    // The ASHRAE 55 regression of indoor clothing against the 6 am outdoor temperature.
    for tout in [-10.0, 0.0, 10.0, 20.0, 30.0] {
        println!(
            "  {tout:>4.0}°C outdoors -> {:.2} clo",
            clo_tout(Temperature::from_celsius(tout)).as_clo()
        );
    }
}
