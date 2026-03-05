//! Thermal comfort models

pub mod adaptive;
pub mod cooling_effect;
pub mod heat_index_lu;
pub mod pet;
pub mod phs;
pub mod pmv;
pub mod pmv_typed;
pub mod ridge_regression;
pub mod set_tmp;
pub mod solar_gain;
pub mod specialty;
pub mod thermal_indices;
pub mod two_nodes_gagge;
pub mod use_fans_heatwaves;
pub mod utci;
pub mod wbgt;
pub mod work_capacity;

// Re-export utilities that are also exposed as models in Python
pub use crate::utilities::clo_tout;

// Re-export commonly used models
pub use adaptive::{
    AdaptiveAshraeResult, AdaptiveEnResult, AdaptiveOptions, adaptive_ashrae, adaptive_en,
};
pub use cooling_effect::{CoolingEffectOptions, cooling_effect};
pub use heat_index_lu::heat_index_lu;
pub use pet::{PetOptions, PetResult, Posture as PetPosture, pet_steady};
pub use phs::{Iso7933Model, PhsOptions, PhsPosture, PhsResult, phs};
pub use pmv::{PmvPpdResult, pmv_a, pmv_athb, pmv_e, pmv_ppd_ashrae, pmv_ppd_iso};
pub use ridge_regression::{
    PredictedBodyTemperatures, RidgeRegressionOptions, Sex as RidgeSex,
    ridge_regression_predict_t_re_t_sk,
};
pub use set_tmp::{SetOptions, set_tmp};
pub use solar_gain::{SolarGainResult, solar_gain};
pub use specialty::{ankle_draft, f_svv, transpose_sharp_altitude, vertical_tmp_grad_ppd};
pub use thermal_indices::{
    at, discomfort_index, esi, heat_index_rothfusz, humidex, humidex_masterson, net, thi, wci,
    wind_chill_temperature,
};
pub use two_nodes_gagge::{
    GaggeTwoNodesJiOptions, GaggeTwoNodesJiResult, GaggeTwoNodesOptions, GaggeTwoNodesResult,
    GaggeTwoNodesSleepOptions, two_nodes_gagge, two_nodes_gagge_ji, two_nodes_gagge_sleep,
};
pub use use_fans_heatwaves::{UseFansHeatwavesResult, use_fans_heatwaves};
pub use utci::{StressCategory, UtciOptions, UtciResult, utci};
pub use wbgt::{WbgtOptions, wbgt};
pub use work_capacity::{
    WorkIntensity, work_capacity_dunne, work_capacity_hothaps, work_capacity_iso,
    work_capacity_niosh,
};
