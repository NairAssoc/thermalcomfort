//! Index maps and coefficient-matrix builders for the JOS3 thermoregulation model.
//!
//! Mirrors `pythermalcomfort/jos3_functions/matrix.py` (pythermalcomfort 4.4.0).
//!
//! JOS3 solves a linear system each timestep over 85 nodes: one "central blood" (CB)
//! node plus up to 7 layers (artery, vein, sfvein, core, muscle, fat, skin) for each of
//! the 17 body segments (not every segment has every layer — see [`index_order`]). This
//! module builds the index maps that fix where each node lives in the 85x85 matrix, and
//! the two blood-flow coefficient matrices (`local_arr`/[`local_arr`],
//! `whole_body`/[`whole_body`]) that are later summed together elsewhere (in
//! `construction.py`, not yet ported) into the full system matrix that gets solved.
//!
//! # Not ported
//!
//! Python's module-level `INDEX`/`VINDEX` dicts (`{layer_name: index_by_layer(layer)}`
//! computed once at import time) are not ported as precomputed statics: `no_std` has no
//! ergonomic lazy-static without pulling in another dependency, and each is a cheap O(17)
//! scan. Callers should call [`index_by_layer`]/[`valid_index_by_layer`] directly at the
//! point of use instead of caching a dict up front.

extern crate alloc;

use alloc::vec::Vec;

use nalgebra::DMatrix;

use super::parameters::NUM_BODY_PARTS;

/// Matrix index (into the 85-node system) of each layer that exists for one body part.
/// `None` means that body part has no such layer (e.g. the torso segments have no
/// `sfvein`, and only head/pelvis have `muscle`/`fat`). Python: one entry of `IDICT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerIndex {
    pub artery: Option<usize>,
    pub vein: Option<usize>,
    pub sfvein: Option<usize>,
    pub core: Option<usize>,
    pub muscle: Option<usize>,
    pub fat: Option<usize>,
    pub skin: Option<usize>,
}

const EMPTY_LAYER_INDEX: LayerIndex = LayerIndex {
    artery: None,
    vein: None,
    sfvein: None,
    core: None,
    muscle: None,
    fat: None,
    skin: None,
};

/// Index (into the 85-node system) of the central blood (CB) node. Python: `IDICT["CB"]`.
pub const CB: usize = 0;

/// Define the index order of the matrix.
///
/// Computed at compile time from a fixed table of which layers exist per body part —
/// this never varies at runtime, unlike Python's `index_order()`/`IDICT` which are
/// (re)computed once at import time from the same fixed table.
///
/// Returns `(idict, num_nodes)`: `idict[i]` is indexed the same way as
/// [`BODY_PART_NAMES`][super::parameters::BODY_PART_NAMES] (`i` = 0 is `head`, ...,
/// `i` = 16 is `right_foot`); `num_nodes` is the total node count including the CB node
/// (Python: `NUM_NODES`).
const fn index_order() -> ([LayerIndex; NUM_BODY_PARTS], usize) {
    // Step 1: which of the 7 layers exist for each body part, in `LAYER_NAMES` order.
    // head/pelvis have every layer except sfvein; neck/chest/back have only
    // artery/vein/core/skin; limb segments have artery/vein/sfvein/core/skin but no
    // muscle/fat. Matches the three loops at the top of Python's `index_order()`.
    let mut exists = [[false; 7]; NUM_BODY_PARTS];
    let mut i = 0;
    while i < NUM_BODY_PARTS {
        exists[i] = if i == 0 || i == 4 {
            // head, pelvis
            [true, true, false, true, true, true, true]
        } else if i == 1 || i == 2 || i == 3 {
            // neck, chest, back
            [true, true, false, true, false, false, true]
        } else {
            // limb segments (left/right shoulder, arm, hand, thigh, leg, foot)
            [true, true, true, true, false, false, true]
        };
        i += 1;
    }

    // Step 2: walk body parts x layers in order, assigning sequential indices starting
    // at 1 (index 0 is reserved for CB). Matches Python's `order_count` loop.
    let mut raw = [[None::<usize>; 7]; NUM_BODY_PARTS];
    let mut order_count: usize = 1;
    let mut bn = 0;
    while bn < NUM_BODY_PARTS {
        let mut ln = 0;
        while ln < 7 {
            if exists[bn][ln] {
                raw[bn][ln] = Some(order_count);
                order_count += 1;
            }
            ln += 1;
        }
        bn += 1;
    }

    let mut idict = [EMPTY_LAYER_INDEX; NUM_BODY_PARTS];
    let mut bn2 = 0;
    while bn2 < NUM_BODY_PARTS {
        idict[bn2] = LayerIndex {
            artery: raw[bn2][0],
            vein: raw[bn2][1],
            sfvein: raw[bn2][2],
            core: raw[bn2][3],
            muscle: raw[bn2][4],
            fat: raw[bn2][5],
            skin: raw[bn2][6],
        };
        bn2 += 1;
    }

    (idict, order_count)
}

const INDEX_ORDER: ([LayerIndex; NUM_BODY_PARTS], usize) = index_order();

/// Matrix index of each layer that exists, per body part (in
/// [`BODY_PART_NAMES`][super::parameters::BODY_PART_NAMES] order). Python: `IDICT`
/// (minus the `"CB"` entry, which is [`CB`] here).
pub const IDICT: [LayerIndex; NUM_BODY_PARTS] = INDEX_ORDER.0;

/// Total node count of the 85-node system (CB + every existing layer of every body
/// part). Python: `NUM_NODES`.
pub const NUM_NODES: usize = INDEX_ORDER.1;

/// Get this layer's field out of one body part's [`LayerIndex`].
fn layer_field(li: &LayerIndex, layer: &str) -> Option<usize> {
    match layer {
        "artery" => li.artery,
        "vein" => li.vein,
        "sfvein" => li.sfvein,
        "core" => li.core,
        "muscle" => li.muscle,
        "fat" => li.fat,
        "skin" => li.skin,
        _ => None,
    }
}

/// Get indices of the matrix by the layer name.
///
/// `layer` is one of `"artery"`, `"vein"`, `"sfvein"`, `"core"`, `"muscle"`, `"fat"`,
/// `"skin"` (case-sensitive; Python lowercases first, callers here should pass
/// already-lowercase names). Python: `index_by_layer`.
pub(crate) fn index_by_layer(layer: &str) -> Vec<usize> {
    let mut out = Vec::with_capacity(NUM_BODY_PARTS);
    for li in IDICT.iter() {
        if let Some(idx) = layer_field(li, layer) {
            out.push(idx);
        }
    }
    out
}

/// Get body-part positions (0..17, into [`BODY_PART_NAMES`][super::parameters::BODY_PART_NAMES])
/// that have this layer. Python: `valid_index_by_layer`.
pub(crate) fn valid_index_by_layer(layer: &str) -> Vec<usize> {
    let mut out = Vec::with_capacity(NUM_BODY_PARTS);
    for (i, li) in IDICT.iter().enumerate() {
        if layer_field(li, layer).is_some() {
            out.push(i);
        }
    }
    out
}

/// Create the matrix of heat exchange coefficients by blood flow within each segment
/// (artery/vein/muscle/fat/skin of the same body part), \[W/K\].
///
/// `1.067` \[Wh/(L*K)\] * blood flow \[L/h\] = \[W/K\]. Python: `local_arr`.
pub(crate) fn local_arr(
    bf_core: &[f64; NUM_BODY_PARTS],
    bf_muscle: &[f64; NUM_BODY_PARTS],
    bf_fat: &[f64; NUM_BODY_PARTS],
    bf_skin: &[f64; NUM_BODY_PARTS],
    bf_ava_hand: f64,
    bf_ava_foot: f64,
) -> DMatrix<f64> {
    const COEFF: f64 = 1.067;
    let mut bf_local = DMatrix::<f64>::zeros(NUM_NODES, NUM_NODES);

    for (i, index_of) in IDICT.iter().enumerate() {
        // These fields always exist for every body part.
        let artery = index_of.artery.expect("every body part has an artery node");
        let vein = index_of.vein.expect("every body part has a vein node");
        let core = index_of.core.expect("every body part has a core node");
        let skin = index_of.skin.expect("every body part has a skin node");

        // Common
        bf_local[(core, artery)] = COEFF * bf_core[i]; // art to cr
        bf_local[(skin, artery)] = COEFF * bf_skin[i]; // art to sk
        bf_local[(vein, core)] = COEFF * bf_core[i]; // vein to cr
        bf_local[(vein, skin)] = COEFF * bf_skin[i]; // vein to sk

        // If the segment has a muscle or fat layer
        if let Some(muscle) = index_of.muscle {
            bf_local[(muscle, artery)] = COEFF * bf_muscle[i]; // art to ms
            bf_local[(vein, muscle)] = COEFF * bf_muscle[i]; // vein to ms
        }
        if let Some(fat) = index_of.fat {
            bf_local[(fat, artery)] = COEFF * bf_fat[i]; // art to fat
            bf_local[(vein, fat)] = COEFF * bf_fat[i]; // vein to fat
        }

        // Only hand
        if i == 7 || i == 10 {
            let sfvein = index_of.sfvein.expect("hands have an sfvein node");
            bf_local[(sfvein, artery)] = COEFF * bf_ava_hand; // art to sfvein
        }
        // Only foot
        if i == 13 || i == 16 {
            let sfvein = index_of.sfvein.expect("feet have an sfvein node");
            bf_local[(sfvein, artery)] = COEFF * bf_ava_foot; // art to sfvein
        }
    }

    bf_local
}

/// Sum `xbf[lo..=hi]` (inclusive), where `xbf` is one per-body-part array.
fn sum_range(xbf: &[f64; NUM_BODY_PARTS], lo: usize, hi_inclusive: usize) -> f64 {
    xbf[lo..=hi_inclusive].iter().sum()
}

/// Get artery and vein blood flow rate \[L/h\] for each body part. Python:
/// `vessel_blood_flow`.
pub(crate) fn vessel_blood_flow(
    bf_core: &[f64; NUM_BODY_PARTS],
    bf_muscle: &[f64; NUM_BODY_PARTS],
    bf_fat: &[f64; NUM_BODY_PARTS],
    bf_skin: &[f64; NUM_BODY_PARTS],
    bf_ava_hand: f64,
    bf_ava_foot: f64,
) -> ([f64; NUM_BODY_PARTS], [f64; NUM_BODY_PARTS]) {
    let mut xbf = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        xbf[i] = bf_core[i] + bf_muscle[i] + bf_fat[i] + bf_skin[i];
    }

    let mut bf_art = [0.0; NUM_BODY_PARTS];
    let mut bf_vein = [0.0; NUM_BODY_PARTS];

    // head
    bf_art[0] = xbf[0];
    bf_vein[0] = xbf[0];

    // neck (+head)
    bf_art[1] = xbf[1] + xbf[0];
    bf_vein[1] = xbf[1] + xbf[0];

    // chest
    bf_art[2] = xbf[2];
    bf_vein[2] = xbf[2];

    // back
    bf_art[3] = xbf[3];
    bf_vein[3] = xbf[3];

    // pelvis (+Thighs, Legs, Feet, AVA_Feet)
    bf_art[4] = xbf[4] + sum_range(&xbf, 11, NUM_BODY_PARTS - 1) + 2.0 * bf_ava_foot;
    bf_vein[4] = xbf[4] + sum_range(&xbf, 11, NUM_BODY_PARTS - 1) + 2.0 * bf_ava_foot;

    // L.Shoulder (+Arm, Hand, (artery only) AVA_Hand)
    bf_art[5] = sum_range(&xbf, 5, 7) + bf_ava_hand;
    bf_vein[5] = sum_range(&xbf, 5, 7);

    // L.Arm (+Hand)
    bf_art[6] = sum_range(&xbf, 6, 7) + bf_ava_hand;
    bf_vein[6] = sum_range(&xbf, 6, 7);

    // L.Hand
    bf_art[7] = xbf[7] + bf_ava_hand;
    bf_vein[7] = xbf[7];

    // R.Shoulder (+Arm, Hand, (artery only) AVA_Hand)
    bf_art[8] = sum_range(&xbf, 8, 10) + bf_ava_hand;
    bf_vein[8] = sum_range(&xbf, 8, 10);

    // R.Arm (+Hand)
    bf_art[9] = sum_range(&xbf, 9, 10) + bf_ava_hand;
    bf_vein[9] = sum_range(&xbf, 9, 10);

    // R.Hand
    bf_art[10] = xbf[10] + bf_ava_hand;
    bf_vein[10] = xbf[10];

    // L.Thigh (+Leg, Foot, (artery only) AVA_Foot)
    bf_art[11] = sum_range(&xbf, 11, 13) + bf_ava_foot;
    bf_vein[11] = sum_range(&xbf, 11, 13);

    // L.Leg (+Foot)
    bf_art[12] = sum_range(&xbf, 12, 13) + bf_ava_foot;
    bf_vein[12] = sum_range(&xbf, 12, 13);

    // L.Foot
    bf_art[13] = xbf[13] + bf_ava_foot;
    bf_vein[13] = xbf[13];

    // R.Thigh (+Leg, Foot, (artery only) AVA_Foot)
    bf_art[14] = sum_range(&xbf, 14, NUM_BODY_PARTS - 1) + bf_ava_foot;
    bf_vein[14] = sum_range(&xbf, 14, NUM_BODY_PARTS - 1);

    // R.Leg (+Foot)
    bf_art[15] = sum_range(&xbf, 15, NUM_BODY_PARTS - 1) + bf_ava_foot;
    bf_vein[15] = sum_range(&xbf, 15, NUM_BODY_PARTS - 1);

    // R.Foot
    bf_art[16] = xbf[16] + bf_ava_foot;
    bf_vein[16] = xbf[16];

    (bf_art, bf_vein)
}

/// Add a single directed blood-flow edge (`up` -> `down`) into `arr`, \[W/K\].
///
/// Coefficient = 1.067 \[Wh/(L*K)\], converting flow \[L/h\] to \[W/K\]. Equivalent to
/// Python's local `flow()` closure inside `whole_body`, except it accumulates directly
/// into the caller's matrix instead of allocating and adding a full zero matrix.
fn add_flow(arr: &mut DMatrix<f64>, up: usize, down: usize, bloodflow: f64) {
    arr[(down, up)] += 1.067 * bloodflow;
}

/// Create the matrix of heat exchange coefficients by blood flow between segments,
/// \[W/K\]. Python: `whole_body`.
pub(crate) fn whole_body(
    bf_art: &[f64; NUM_BODY_PARTS],
    bf_vein: &[f64; NUM_BODY_PARTS],
    bf_ava_hand: f64,
    bf_ava_foot: f64,
) -> DMatrix<f64> {
    // Matrix offsets of segments (artery-layer index of each body part; body-part
    // positions below match `BODY_PART_NAMES`/`IDICT` order: 0=head, 1=neck, 2=chest,
    // 3=back, 4=pelvis, 5=left_shoulder, 6=left_arm, 7=left_hand, 8=right_shoulder,
    // 9=right_arm, 10=right_hand, 11=left_thigh, 12=left_leg, 13=left_foot,
    // 14=right_thigh, 15=right_leg, 16=right_foot).
    let head = IDICT[0].artery.unwrap();
    let neck = IDICT[1].artery.unwrap();
    let chest = IDICT[2].artery.unwrap();
    let back = IDICT[3].artery.unwrap();
    let pelvis = IDICT[4].artery.unwrap();
    let left_shoulder = IDICT[5].artery.unwrap();
    let left_arm = IDICT[6].artery.unwrap();
    let left_hand = IDICT[7].artery.unwrap();
    let right_shoulder = IDICT[8].artery.unwrap();
    let right_arm = IDICT[9].artery.unwrap();
    let right_hand = IDICT[10].artery.unwrap();
    let left_thigh = IDICT[11].artery.unwrap();
    let left_leg = IDICT[12].artery.unwrap();
    let left_foot = IDICT[13].artery.unwrap();
    let right_thigh = IDICT[14].artery.unwrap();
    let right_leg = IDICT[15].artery.unwrap();
    let right_foot = IDICT[16].artery.unwrap();

    let mut arr83 = DMatrix::<f64>::zeros(NUM_NODES, NUM_NODES);

    add_flow(&mut arr83, CB, neck, bf_art[1]); // CB to neck.art
    add_flow(&mut arr83, neck, head, bf_art[0]); // neck.art to head.art
    add_flow(&mut arr83, head + 1, neck + 1, bf_vein[0]); // head.vein to neck.vein
    add_flow(&mut arr83, neck + 1, CB, bf_vein[1]); // neck.vein to CB

    add_flow(&mut arr83, CB, chest, bf_art[2]); // CB to chest.art
    add_flow(&mut arr83, chest + 1, CB, bf_vein[2]); // chest.vein to CB

    add_flow(&mut arr83, CB, back, bf_art[3]); // CB to back.art
    add_flow(&mut arr83, back + 1, CB, bf_vein[3]); // back.vein to CB

    add_flow(&mut arr83, CB, pelvis, bf_art[4]); // CB to pelvis.art
    add_flow(&mut arr83, pelvis + 1, CB, bf_vein[4]); // pelvis.vein to CB

    add_flow(&mut arr83, CB, left_shoulder, bf_art[5]); // CB to left_shoulder.art
    add_flow(&mut arr83, left_shoulder, left_arm, bf_art[6]); // left_shoulder.art to left_arm.art
    add_flow(&mut arr83, left_arm, left_hand, bf_art[7]); // left_arm.art to left_hand.art
    add_flow(&mut arr83, left_hand + 1, left_arm + 1, bf_vein[7]); // left_hand.vein to left_arm.vein
    add_flow(&mut arr83, left_arm + 1, left_shoulder + 1, bf_vein[6]); // left_arm.vein to left_shoulder.vein
    add_flow(&mut arr83, left_shoulder + 1, CB, bf_vein[5]); // left_shoulder.vein to CB
    add_flow(&mut arr83, left_hand + 2, left_arm + 2, bf_ava_hand); // left_hand.sfvein to left_arm.sfvein
    add_flow(&mut arr83, left_arm + 2, left_shoulder + 2, bf_ava_hand); // left_arm.sfvein to left_shoulder.sfvein
    add_flow(&mut arr83, left_shoulder + 2, CB, bf_ava_hand); // left_shoulder.sfvein to CB

    add_flow(&mut arr83, CB, right_shoulder, bf_art[8]); // CB to right_shoulder.art
    add_flow(&mut arr83, right_shoulder, right_arm, bf_art[9]); // right_shoulder.art to right_arm.art
    add_flow(&mut arr83, right_arm, right_hand, bf_art[10]); // right_arm.art to right_hand.art
    add_flow(&mut arr83, right_hand + 1, right_arm + 1, bf_vein[10]); // right_hand.vein to right_arm.vein
    add_flow(&mut arr83, right_arm + 1, right_shoulder + 1, bf_vein[9]); // right_arm.vein to right_shoulder.vein
    add_flow(&mut arr83, right_shoulder + 1, CB, bf_vein[8]); // right_shoulder.vein to CB
    add_flow(&mut arr83, right_hand + 2, right_arm + 2, bf_ava_hand); // right_hand.sfvein to right_arm.sfvein
    add_flow(&mut arr83, right_arm + 2, right_shoulder + 2, bf_ava_hand); // right_arm.sfvein to right_shoulder.sfvein
    add_flow(&mut arr83, right_shoulder + 2, CB, bf_ava_hand); // right_shoulder.sfvein to CB

    add_flow(&mut arr83, pelvis, left_thigh, bf_art[11]); // pelvis to left_thigh.art
    add_flow(&mut arr83, left_thigh, left_leg, bf_art[12]); // left_thigh.art to left_leg.art
    add_flow(&mut arr83, left_leg, left_foot, bf_art[13]); // left_leg.art to left_foot.art
    add_flow(&mut arr83, left_foot + 1, left_leg + 1, bf_vein[13]); // left_foot.vein to left_leg.vein
    add_flow(&mut arr83, left_leg + 1, left_thigh + 1, bf_vein[12]); // left_leg.vein to left_thigh.vein
    add_flow(&mut arr83, left_thigh + 1, pelvis + 1, bf_vein[11]); // left_thigh.vein to pelvis
    add_flow(&mut arr83, left_foot + 2, left_leg + 2, bf_ava_foot); // left_foot.sfvein to left_leg.sfvein
    add_flow(&mut arr83, left_leg + 2, left_thigh + 2, bf_ava_foot); // left_leg.sfvein to left_thigh.sfvein
    add_flow(&mut arr83, left_thigh + 2, pelvis + 1, bf_ava_foot); // left_thigh.sfvein to pelvis (Python labels this "left_thigh.vein" but uses the sfvein offset; ported verbatim)

    add_flow(&mut arr83, pelvis, right_thigh, bf_art[14]); // pelvis to right_thigh.art
    add_flow(&mut arr83, right_thigh, right_leg, bf_art[15]); // right_thigh.art to right_leg.art
    add_flow(&mut arr83, right_leg, right_foot, bf_art[16]); // right_leg.art to right_foot.art
    add_flow(&mut arr83, right_foot + 1, right_leg + 1, bf_vein[16]); // right_foot.vein to right_leg.vein
    add_flow(&mut arr83, right_leg + 1, right_thigh + 1, bf_vein[15]); // right_leg.vein to right_thigh.vein
    add_flow(&mut arr83, right_thigh + 1, pelvis + 1, bf_vein[14]); // right_thigh.vein to pelvis
    add_flow(&mut arr83, right_foot + 2, right_leg + 2, bf_ava_foot); // right_foot.sfvein to right_leg.sfvein
    add_flow(&mut arr83, right_leg + 2, right_thigh + 2, bf_ava_foot); // right_leg.sfvein to right_thigh.sfvein
    add_flow(&mut arr83, right_thigh + 2, pelvis + 1, bf_ava_foot); // right_thigh.sfvein to pelvis

    arr83
}

#[cfg(test)]
// Reference values are pasted verbatim from the Python oracle, and rustfmt groups their
// fractional digits but not the integer part. Regrouping by hand to satisfy the lint would
// mean editing numbers whose whole value is that they were not edited.
#[allow(clippy::inconsistent_digit_grouping)]
mod tests {
    use super::*;

    /// Cross-check against `pythermalcomfort.jos3_functions.matrix.NUM_NODES` (85),
    /// obtained by running the reference Python module directly.
    #[test]
    fn num_nodes_matches_python() {
        assert_eq!(NUM_NODES, 85);
    }

    /// Cross-check a handful of `IDICT` entries against the Python dict, e.g.
    /// `IDICT["head"] = {'artery': 1, 'vein': 2, 'sfvein': None, 'core': 3, 'muscle': 4,
    /// 'fat': 5, 'skin': 6}` and
    /// `IDICT["right_foot"] = {'artery': 80, 'vein': 81, 'sfvein': 82, 'core': 83,
    /// 'muscle': None, 'fat': None, 'skin': 84}`.
    #[test]
    fn idict_matches_python() {
        let head = IDICT[0]; // BODY_PART_NAMES[0] == "head"
        assert_eq!(head.artery, Some(1));
        assert_eq!(head.vein, Some(2));
        assert_eq!(head.sfvein, None);
        assert_eq!(head.core, Some(3));
        assert_eq!(head.muscle, Some(4));
        assert_eq!(head.fat, Some(5));
        assert_eq!(head.skin, Some(6));

        let neck = IDICT[1];
        assert_eq!(neck.artery, Some(7));
        assert_eq!(neck.vein, Some(8));
        assert_eq!(neck.sfvein, None);
        assert_eq!(neck.core, Some(9));
        assert_eq!(neck.muscle, None);
        assert_eq!(neck.fat, None);
        assert_eq!(neck.skin, Some(10));

        let pelvis = IDICT[4];
        assert_eq!(pelvis.artery, Some(19));
        assert_eq!(pelvis.skin, Some(24));

        let left_hand = IDICT[7];
        assert_eq!(left_hand.artery, Some(35));
        assert_eq!(left_hand.sfvein, Some(37));

        let right_foot = IDICT[16]; // last body part
        assert_eq!(right_foot.artery, Some(80));
        assert_eq!(right_foot.vein, Some(81));
        assert_eq!(right_foot.sfvein, Some(82));
        assert_eq!(right_foot.core, Some(83));
        assert_eq!(right_foot.muscle, None);
        assert_eq!(right_foot.fat, None);
        assert_eq!(right_foot.skin, Some(84));
    }

    /// Cross-check against Python's `INDEX`/`VINDEX` dicts for a couple of layers.
    #[test]
    fn index_by_layer_matches_python() {
        assert_eq!(index_by_layer("muscle"), alloc::vec![4, 22]);
        assert_eq!(index_by_layer("fat"), alloc::vec![5, 23]);
        assert_eq!(
            index_by_layer("sfvein"),
            alloc::vec![27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82]
        );
        assert_eq!(valid_index_by_layer("muscle"), alloc::vec![0, 4]);
        assert_eq!(
            valid_index_by_layer("sfvein"),
            alloc::vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    /// Cross-check against a run of the reference Python `local_arr` with
    /// `bf_core[i] = 100+i`, `bf_muscle[i] = 200+i`, `bf_fat[i] = 300+i`,
    /// `bf_skin[i] = 400+i` (i = 0..17), `bf_ava_hand = 555`, `bf_ava_foot = 777`:
    /// `np.nonzero(bf_local)` found exactly 80 nonzero cells, including
    /// `[2,3] = 106.69999999999999`, `[37,35] = 592.185` (left hand sfvein<-artery,
    /// AVA_hand path) and `[67,65] = 829.059` (left foot sfvein<-artery, AVA_foot path).
    #[test]
    fn local_arr_matches_python() {
        let mut bf_core = [0.0; NUM_BODY_PARTS];
        let mut bf_muscle = [0.0; NUM_BODY_PARTS];
        let mut bf_fat = [0.0; NUM_BODY_PARTS];
        let mut bf_skin = [0.0; NUM_BODY_PARTS];
        for i in 0..NUM_BODY_PARTS {
            bf_core[i] = 100.0 + i as f64;
            bf_muscle[i] = 200.0 + i as f64;
            bf_fat[i] = 300.0 + i as f64;
            bf_skin[i] = 400.0 + i as f64;
        }
        let bf_ava_hand = 555.0;
        let bf_ava_foot = 777.0;

        let m = local_arr(
            &bf_core,
            &bf_muscle,
            &bf_fat,
            &bf_skin,
            bf_ava_hand,
            bf_ava_foot,
        );
        assert_eq!(m.nrows(), 85);
        assert_eq!(m.ncols(), 85);

        let nonzero_count = m.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero_count, 80);

        assert!((m[(2, 3)] - 106.699_999_999_999_99).abs() < 1e-9);
        assert!((m[(37, 35)] - 592.185).abs() < 1e-9); // left_hand sfvein <- artery (AVA hand)
        assert!((m[(67, 65)] - 829.059).abs() < 1e-9); // left_foot sfvein <- artery (AVA foot)
        assert!((m[(84, 80)] - 443.871_999_999_999_96).abs() < 1e-9); // right_foot vein <- skin
    }

    /// Cross-check against the same reference Python run's `vessel_blood_flow` output:
    /// `bf_art = [1000.0, 2004.0, 1008.0, 1012.0, 8894.0, 3627.0, 2607.0, 1583.0, 3663.0,
    /// 2631.0, 1595.0, 3921.0, 2877.0, 1829.0, 3957.0, 2901.0, 1841.0]`,
    /// `bf_vein = [1000.0, 2004.0, 1008.0, 1012.0, 8894.0, 3072.0, 2052.0, 1028.0, 3108.0,
    /// 2076.0, 1040.0, 3144.0, 2100.0, 1052.0, 3180.0, 2124.0, 1064.0]`.
    #[test]
    fn vessel_blood_flow_matches_python() {
        let mut bf_core = [0.0; NUM_BODY_PARTS];
        let mut bf_muscle = [0.0; NUM_BODY_PARTS];
        let mut bf_fat = [0.0; NUM_BODY_PARTS];
        let mut bf_skin = [0.0; NUM_BODY_PARTS];
        for i in 0..NUM_BODY_PARTS {
            bf_core[i] = 100.0 + i as f64;
            bf_muscle[i] = 200.0 + i as f64;
            bf_fat[i] = 300.0 + i as f64;
            bf_skin[i] = 400.0 + i as f64;
        }
        let bf_ava_hand = 555.0;
        let bf_ava_foot = 777.0;

        let (bf_art, bf_vein) = vessel_blood_flow(
            &bf_core,
            &bf_muscle,
            &bf_fat,
            &bf_skin,
            bf_ava_hand,
            bf_ava_foot,
        );

        let expected_art = [
            1000.0, 2004.0, 1008.0, 1012.0, 8894.0, 3627.0, 2607.0, 1583.0, 3663.0, 2631.0, 1595.0,
            3921.0, 2877.0, 1829.0, 3957.0, 2901.0, 1841.0,
        ];
        let expected_vein = [
            1000.0, 2004.0, 1008.0, 1012.0, 8894.0, 3072.0, 2052.0, 1028.0, 3108.0, 2076.0, 1040.0,
            3144.0, 2100.0, 1052.0, 3180.0, 2124.0, 1064.0,
        ];
        for i in 0..NUM_BODY_PARTS {
            assert!((bf_art[i] - expected_art[i]).abs() < 1e-9, "bf_art[{i}]");
            assert!((bf_vein[i] - expected_vein[i]).abs() < 1e-9, "bf_vein[{i}]");
        }
    }

    /// Cross-check against the reference Python `whole_body(bf_art, bf_vein, 555.0,
    /// 777.0)` fed the `bf_art`/`bf_vein` above: `np.nonzero(arr83)` found exactly 46
    /// nonzero cells, including `[0,8] = 2138.268` (neck.vein -> CB) and
    /// `[55,19] = 4183.706999999999` (pelvis -> left_thigh.art).
    #[test]
    fn whole_body_matches_python() {
        let bf_art = [
            1000.0, 2004.0, 1008.0, 1012.0, 8894.0, 3627.0, 2607.0, 1583.0, 3663.0, 2631.0, 1595.0,
            3921.0, 2877.0, 1829.0, 3957.0, 2901.0, 1841.0,
        ];
        let bf_vein = [
            1000.0, 2004.0, 1008.0, 1012.0, 8894.0, 3072.0, 2052.0, 1028.0, 3108.0, 2076.0, 1040.0,
            3144.0, 2100.0, 1052.0, 3180.0, 2124.0, 1064.0,
        ];
        let bf_ava_hand = 555.0;
        let bf_ava_foot = 777.0;

        let m = whole_body(&bf_art, &bf_vein, bf_ava_hand, bf_ava_foot);
        assert_eq!(m.nrows(), 85);
        assert_eq!(m.ncols(), 85);

        let nonzero_count = m.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero_count, 46);

        assert!((m[(0, 8)] - 2138.268).abs() < 1e-9); // neck.vein -> CB
        assert!((m[(55, 19)] - 4183.706_999_999_999).abs() < 1e-6); // pelvis -> left_thigh.art
        assert!((m[(1, 7)] - 1067.0).abs() < 1e-9); // CB -> neck.art
        assert!((m[(0, 27)] - 592.185).abs() < 1e-9); // left_shoulder.sfvein -> CB (AVA hand)
    }
}
