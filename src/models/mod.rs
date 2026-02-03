//! Thermal comfort models

pub mod pmv;
pub mod pmv_typed;
pub mod two_nodes_gagge;
pub mod set_tmp;
pub mod cooling_effect;
pub mod utci;
pub mod adaptive;
pub mod wbgt;
pub mod thermal_indices;
pub mod work_capacity;
pub mod specialty;
pub mod solar_gain;
pub mod use_fans_heatwaves;
pub mod heat_index_lu;

// Re-export commonly used models
pub use pmv::{pmv_ppd_iso, pmv_ppd_ashrae, pmv_a, pmv_e, pmv_athb, PmvPpdResult};
pub use two_nodes_gagge::{two_nodes_gagge, GaggeTwoNodesResult, GaggeTwoNodesOptions};
pub use set_tmp::{set_tmp, SetOptions};
pub use cooling_effect::{cooling_effect, CoolingEffectOptions};
pub use utci::{utci, UtciResult, UtciOptions, StressCategory};
pub use adaptive::{adaptive_ashrae, adaptive_en, AdaptiveAshraeResult, AdaptiveEnResult, AdaptiveOptions};
pub use wbgt::{wbgt, WbgtOptions};
pub use thermal_indices::{
    wci, wind_chill_temperature, humidex, humidex_masterson,
    thi, discomfort_index, heat_index_rothfusz, at, net, esi
};
pub use work_capacity::{
    work_capacity_iso, work_capacity_niosh, work_capacity_dunne,
    work_capacity_hothaps, WorkIntensity
};
pub use specialty::{ankle_draft, vertical_tmp_grad_ppd, f_svv, transpose_sharp_altitude};
pub use solar_gain::{solar_gain, SolarGainResult};
pub use use_fans_heatwaves::{use_fans_heatwaves, UseFansHeatwavesResult};
pub use heat_index_lu::heat_index_lu;
