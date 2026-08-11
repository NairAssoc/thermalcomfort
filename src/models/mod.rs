//! Thermal comfort models

pub mod adaptive;
pub mod cooling_effect;
pub mod heat_index_lu;
pub mod ireq;
pub mod jos3;
pub mod pet;
pub mod phs;
pub mod pmv;
pub mod pmv_typed;
pub mod ridge_regression;
pub mod set_tmp;
pub mod solar_gain;
pub mod specialty;
pub mod sports_heat_stress_risk;
pub mod thermal_indices;
pub mod two_nodes_gagge;
pub mod two_nodes_gagge_sleep;
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
pub use cooling_effect::{CoolingEffectInputs, CoolingEffectOptions, cooling_effect};
pub use heat_index_lu::{HeatIndexLuInputs, HeatIndexLuOptions, heat_index_lu};
pub use ireq::{DurationLimitedExposure, IreqInputs, IreqOptions, IreqResult, ireq};
pub use pet::{PetInputs, PetOptions, PetResult, Posture as PetPosture, pet_steady};
pub use phs::{Iso7933Model, PhsInputs, PhsOptions, PhsPosture, PhsResult, phs};
pub use pmv::{
    Iso7730Model, PmvAInputs, PmvAOptions, PmvAthbInputs, PmvAthbOptions, PmvEInputs, PmvEOptions,
    PmvPpdAshraeOptions, PmvPpdInputs, PmvPpdIsoOptions, PmvPpdResult, pmv_a, pmv_athb, pmv_e,
    pmv_ppd_ashrae, pmv_ppd_iso,
};
pub use ridge_regression::{
    PredictedBodyTemperatures, RidgeRegressionInputs, RidgeRegressionOptions,
    ridge_regression_predict_t_re_t_sk,
};
pub use set_tmp::{SetInputs, SetOptions, set_tmp};
pub use solar_gain::{SolarGainInputs, SolarGainOptions, SolarGainResult, solar_gain};
pub use specialty::{
    AnkleDraftInputs, AnkleDraftOptions, FSvvInputs, VerticalTmpGradPpdInputs,
    VerticalTmpGradPpdOptions, ankle_draft, f_svv, transpose_sharp_altitude, vertical_tmp_grad_ppd,
};
pub use sports_heat_stress_risk::{
    Sports, SportsHeatStressRisk, SportsHeatStressRiskInputs, SportsValues, sports_heat_stress_risk,
};
pub use thermal_indices::{
    AtInputs, AtOptions, DiscomfortCondition, DiscomfortIndexInputs, DiscomfortIndexResult,
    EsiInputs, EsiOptions, HeatIndexResult, HeatIndexRothfuszInputs, HeatIndexRothfuszOptions,
    HeatIndexSchoenInputs, HeatIndexSchoenOptions, HeatIndexStress, HumidexDiscomfort,
    HumidexInputs, HumidexModel, HumidexOptions, HumidexResult, NetInputs, NetOptions, ThiInputs,
    ThiOptions, WciInputs, WciOptions, WindChillTemperatureInputs, WindChillTemperatureOptions, at,
    discomfort_index, esi, heat_index_rothfusz, heat_index_schoen, humidex, net, thi, wci,
    wind_chill_temperature,
};
pub use two_nodes_gagge::{
    GaggeTwoNodesInputs, GaggeTwoNodesJiInputs, GaggeTwoNodesJiOptions, GaggeTwoNodesJiResult,
    GaggeTwoNodesOptions, GaggeTwoNodesResult, two_nodes_gagge, two_nodes_gagge_ji,
};
pub use two_nodes_gagge_sleep::{
    GaggeTwoNodesSleepOptions, GaggeTwoNodesSleepResult, MismatchedScheduleLengths, SleepInputs,
    two_nodes_gagge_sleep,
};
pub use use_fans_heatwaves::{
    UseFansHeatwavesInputs, UseFansHeatwavesOptions, UseFansHeatwavesResult, use_fans_heatwaves,
};
pub use utci::{StressCategory, UtciInputs, UtciOptions, UtciResult, utci};
pub use wbgt::{WbgtInputs, WbgtOptions, wbgt};
pub use work_capacity::{
    WorkCapacityIntensityOptions, WorkIntensity, work_capacity_dunne, work_capacity_hothaps,
    work_capacity_iso, work_capacity_niosh,
};
