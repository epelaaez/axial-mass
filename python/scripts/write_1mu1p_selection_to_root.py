"""
Add 1mu1p selection and truth-classification branches to a ROOT event tree.

Usage
-----
    python write_1mu1p_selection_to_root.py input.root [output.root] [options]

If output.root is omitted the script writes to
<input_stem>_1mu1p.root in the same directory.
"""

import argparse
import os
import shutil
from enum import IntEnum

import awkward as ak
import numpy as np
import uproot

from STVTools import STVTools


PROTON_MASS          = 0.938272  # GeV/c^2
TRACK_SCORE_CUT      = 0.5
PROTON_LLR_PID_SCORE = 0.05

FVX = 256.0
FVY = 232.0
FVZ = 1037.0
BORDERX = 10.0
BORDERY = 10.0
BORDERZ = 10.0


class TruthCategory(IntEnum):
    BACKGROUND = 0
    SIGNAL_1MU1P = 1


TRUTH_CATEGORY_NAMES = {
    TruthCategory.BACKGROUND: "background",
    TruthCategory.SIGNAL_1MU1P: "signal_1mu1p",
}

STV_RETURN_METHODS = [
    "ReturnkMiss",
    "ReturnEMiss",
    "ReturnPMissMinus",
    "ReturnPMiss",
    "ReturnPt",
    "ReturnPtx",
    "ReturnPty",
    "ReturnPnPerp",
    "ReturnPnPerpx",
    "ReturnPnPerpy",
    "ReturnPnPar",
    "ReturnPL",
    "ReturnPn",
    "ReturnDeltaAlphaT",
    "ReturnDeltaAlpha3Dq",
    "ReturnDeltaAlpha3DMu",
    "ReturnDeltaPhiT",
    "ReturnDeltaPhi3D",
    "ReturnECal",
    "ReturnECalMB",
    "ReturnEQE",
    "ReturnQ2",
    "ReturnA",
]

STV_BRANCH_NAMES = {
    method_name: method_name.replace("Return", "", 1)
    for method_name in STV_RETURN_METHODS
}

INVALID_STV_VALUE = -9999.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add a reco 1mu1p selection/STV tree to a ROOT file."
    )
    parser.add_argument("src", help="Input ROOT file")
    parser.add_argument(
        "dst",
        nargs="?",
        default=None,
        help="Output ROOT file (default: <src_stem>_1mu1p.root)",
    )
    parser.add_argument(
        "--selection-tree",
        default="nuselection/NeutrinoSelectionFilter",
        help="Tree holding event, reco, and truth particle branches "
        "(default: nuselection/NeutrinoSelectionFilter)",
    )
    parser.add_argument(
        "--eval-tree",
        default="wcpselection/T_eval",
        help="Tree holding truth_isCC and event weights (default: wcpselection/T_eval)",
    )
    parser.add_argument(
        "--bdt-tree",
        default="wcpselection/T_BDTvars",
        help="Tree holding numu_score (default: wcpselection/T_BDTvars)",
    )
    parser.add_argument(
        "--pfeval-tree",
        default="wcpselection/T_PFeval",
        help="Tree holding reco_muonMomentum (default: wcpselection/T_PFeval)",
    )
    parser.add_argument(
        "--kine-tree",
        default="wcpselection/T_KINEvars",
        help="Tree holding kine_reco_Enu (default: wcpselection/T_KINEvars)",
    )
    parser.add_argument(
        "--output-tree",
        default="selection1mu1p",
        help="New tree to write selection/STV branches to (default: selection1mu1p)",
    )
    parser.add_argument(
        "--selected-branch",
        default="selected_1mu1p",
        help="Name of reco-selection bool branch to write (default: selected_1mu1p)",
    )
    parser.add_argument(
        "--truth-branch",
        default="true_1mu1p",
        help="Name of truth-signal bool branch to write (default: true_1mu1p)",
    )
    parser.add_argument(
        "--truth-category-branch",
        default="truth_1mu1p_category",
        help="Name of integer truth category branch to write "
        "(0=background, 1=signal_1mu1p by default)",
    )
    parser.add_argument(
        "--skip-truth-categories",
        action="store_true",
        help="Write reco selection and STV branches only. Do not read truth branches or write truth labels.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Only evaluate the first N events. Remaining events are left unselected.",
    )
    return parser.parse_args()


def in_fv(x, y, z):
    return (
        x < (FVX - BORDERX)
        and x > BORDERX
        and y < (FVY / 2.0 - BORDERY)
        and y > (-FVY / 2.0 + BORDERY)
        and z < (FVZ - BORDERZ)
        and z > BORDERZ
    )


def spherical_to_cartesian(mag, theta, phi):
    return np.array(
        [
            mag * np.sin(theta) * np.cos(phi),
            mag * np.sin(theta) * np.sin(phi),
            mag * np.cos(theta),
        ]
    )


def is_meson_or_antimeson(pdg):
    pdg = np.abs(pdg)
    return (
        (pdg < 9900000)
        & ((pdg // 1000) % 10 == 0)
        & ((pdg // 100) % 10 != 0)
        & ~((901 <= pdg) & (pdg <= 930))
        & (pdg != 110)
        & (pdg != 990)
        & (pdg != 998)
        & (pdg != 999)
        & (pdg != 100)
    )


def truth_categories(selection_tree, eval_tree, n_events, entry_stop=None):
    print("  Loading truth branches from selection tree...", flush=True)
    selection_arrays = selection_tree.arrays(
        ["nu_pdg", "mc_px", "mc_py", "mc_pz", "mc_pdg"],
        entry_stop=entry_stop,
        library="ak",
    )
    print("  Loading truth branches from eval tree...", flush=True)
    eval_arrays = eval_tree.arrays(["truth_isCC"], entry_stop=entry_stop, library="ak")
    print("  Truth branch loading complete.", flush=True)

    nu_pdg = ak.to_numpy(selection_arrays["nu_pdg"])
    truth_is_cc = ak.to_numpy(eval_arrays["truth_isCC"])
    mc_px = selection_arrays["mc_px"]
    mc_py = selection_arrays["mc_py"]
    mc_pz = selection_arrays["mc_pz"]
    mc_pdg = selection_arrays["mc_pdg"]

    p_mag = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2)

    mu_nu_cc = (np.abs(nu_pdg) == 14) & (truth_is_cc == 1)
    one_muon = ak.to_numpy(ak.sum((np.abs(mc_pdg) == 13) & (p_mag > 0.1), axis=1) == 1)
    one_proton = ak.to_numpy(ak.sum((np.abs(mc_pdg) == 2212) & (p_mag > 0.25), axis=1) == 1)
    no_charged_pions = ak.to_numpy(ak.sum((np.abs(mc_pdg) == 211) & (p_mag > 0.07), axis=1) == 0)
    no_neutral_pions = ak.to_numpy(ak.sum(mc_pdg == 111, axis=1) == 0)
    no_heavier_mesons = ak.to_numpy(
        ak.sum(
            (mc_pdg != 111)
            & (np.abs(mc_pdg) != 211)
            & is_meson_or_antimeson(mc_pdg),
            axis=1,
        )
        == 0
    )

    is_signal = (
        mu_nu_cc
        & one_muon
        & one_proton
        & no_charged_pions
        & no_neutral_pions
        & no_heavier_mesons
    )

    true_1mu1p = np.zeros(n_events, dtype=np.bool_)
    truth_category = np.full(n_events, TruthCategory.BACKGROUND, dtype=np.int32)

    n_classified = len(nu_pdg)
    true_1mu1p[:n_classified] = is_signal.astype(np.bool_)
    classified_categories = truth_category[:n_classified]
    classified_categories[is_signal] = TruthCategory.SIGNAL_1MU1P
    return true_1mu1p, truth_category


def load_reco_inputs(selection_tree, eval_tree, bdt_tree, pfeval_tree, kine_tree, entry_stop=None):
    selection_branches = [
        "reco_nu_vtx_sce_x",
        "reco_nu_vtx_sce_y",
        "reco_nu_vtx_sce_z",
        "trk_llr_pid_score_v",
        "n_pfps",
        "pfp_generation_v",
        "trk_score_v",
        "nslice",
        "pfpdg",
        "trk_theta_v",
        "trk_phi_v",
        "trk_sce_start_x_v",
        "trk_sce_start_y_v",
        "trk_sce_start_z_v",
        "trk_sce_end_x_v",
        "trk_sce_end_y_v",
        "trk_sce_end_z_v",
        "trk_range_muon_mom_v",
        "trk_energy_proton_v",
        "trk_mcs_muon_mom_v",
    ]

    print("  Loading reco branches from kine tree...", flush=True)
    kine_arrays = kine_tree.arrays(["kine_reco_Enu"], entry_stop=entry_stop, library="ak")
    print("  Loading reco branches from eval tree...", flush=True)
    eval_arrays = eval_tree.arrays(["match_isFC"], entry_stop=entry_stop, library="ak")
    print("  Loading reco branches from BDT tree...", flush=True)
    bdt_arrays = bdt_tree.arrays(["numu_score"], entry_stop=entry_stop, library="ak")
    print("  Loading reco branches from PFeval tree...", flush=True)
    pfeval_arrays = pfeval_tree.arrays(["reco_muonMomentum"], entry_stop=entry_stop, library="ak")
    print("  Loading reco branches from selection tree...", flush=True)
    selection_arrays = selection_tree.arrays(selection_branches, entry_stop=entry_stop, library="ak")
    print("  Reco branch loading complete.", flush=True)

    return {
        "kine_reco_Enu": kine_arrays["kine_reco_Enu"],
        "match_isFC": eval_arrays["match_isFC"],
        "numu_score": bdt_arrays["numu_score"],
        "reco_muonMomentum": pfeval_arrays["reco_muonMomentum"],
        "reco_nu_vtx_x": selection_arrays["reco_nu_vtx_sce_x"],
        "reco_nu_vtx_y": selection_arrays["reco_nu_vtx_sce_y"],
        "reco_nu_vtx_z": selection_arrays["reco_nu_vtx_sce_z"],
        "trk_llr_pid_score_v": selection_arrays["trk_llr_pid_score_v"],
        "n_pfps": selection_arrays["n_pfps"],
        "pfp_generation_v": selection_arrays["pfp_generation_v"],
        "trk_score_v": selection_arrays["trk_score_v"],
        "nslice": selection_arrays["nslice"],
        "pfpdg": selection_arrays["pfpdg"],
        "trk_theta_v": selection_arrays["trk_theta_v"],
        "trk_phi_v": selection_arrays["trk_phi_v"],
        "trk_sce_start_x_v": selection_arrays["trk_sce_start_x_v"],
        "trk_sce_start_y_v": selection_arrays["trk_sce_start_y_v"],
        "trk_sce_start_z_v": selection_arrays["trk_sce_start_z_v"],
        "trk_sce_end_x_v": selection_arrays["trk_sce_end_x_v"],
        "trk_sce_end_y_v": selection_arrays["trk_sce_end_y_v"],
        "trk_sce_end_z_v": selection_arrays["trk_sce_end_z_v"],
        "trk_range_muon_mom_v": selection_arrays["trk_range_muon_mom_v"],
        "trk_energy_proton_v": selection_arrays["trk_energy_proton_v"],
        "trk_mcs_muon_mom_v": selection_arrays["trk_mcs_muon_mom_v"],
    }


def stv_values(stv_tool):
    return {
        branch_name: float(getattr(stv_tool, method_name)())
        for method_name, branch_name in STV_BRANCH_NAMES.items()
    }


def default_stv_values():
    return {
        branch_name: INVALID_STV_VALUE
        for branch_name in STV_BRANCH_NAMES.values()
    }


def evaluate_reco_event(event_idx, data):
    # Preselection to match original xml: require a reconstructed neutrino energy,
    # fully-contained match, high numu BDT score (reject cosmic), and a valid reco muon object.
    if not (
        (data["kine_reco_Enu"][event_idx] > 0)
        and (data["match_isFC"][event_idx] == 1)
        and (data["numu_score"][event_idx] > 0.9)
        and (data["reco_muonMomentum"][event_idx][3] > 0)
    ):
        return False, None

    candidate_index = []
    reco_shower_count = 0
    reco_track_count = 0

    # Count primary PFParticles only. For this topology we want exactly two
    # track-like primaries and zero shower-like primaries.
    for pfp_idx in range(data["n_pfps"][event_idx]):
        if data["pfp_generation_v"][event_idx][pfp_idx] != 2:
            continue

        if data["trk_score_v"][event_idx][pfp_idx] <= TRACK_SCORE_CUT:
            reco_shower_count += 1
        else:
            reco_track_count += 1
            candidate_index.append(pfp_idx)

    if reco_shower_count != 0 or reco_track_count != 2 or data["nslice"][event_idx] != 1:
        return False, None

    if len(candidate_index) != 2:
        return False, None

    # Assign the more muon-like LLR PID score as the muon
    # candidate and the other track as the proton candidate.
    first_pid_score = data["trk_llr_pid_score_v"][event_idx][candidate_index[0]]
    second_pid_score = data["trk_llr_pid_score_v"][event_idx][candidate_index[1]]

    if first_pid_score > second_pid_score:
        candidate_muon_index = candidate_index[0]
        candidate_proton_index = candidate_index[1]
    else:
        candidate_muon_index = candidate_index[1]
        candidate_proton_index = candidate_index[0]

    candidate_proton_pid_score = data["trk_llr_pid_score_v"][event_idx][candidate_proton_index]

    # Both reconstructed tracks are expected to be track-like
    # muon PFParticles in this ntuple convention.
    if (
        data["pfpdg"][event_idx][candidate_muon_index] != 13
        or data["pfpdg"][event_idx][candidate_proton_index] != 13
    ):
        return False, None

    vertex_vector = np.array(
        [
            data["reco_nu_vtx_x"][event_idx],
            data["reco_nu_vtx_y"][event_idx],
            data["reco_nu_vtx_z"][event_idx],
        ]
    )
    # Keep the reconstructed vertex safely inside the detector active volume.
    if not in_fv(*vertex_vector):
        return False, None

    muon_start_vector = np.array(
        [
            data["trk_sce_start_x_v"][event_idx][candidate_muon_index],
            data["trk_sce_start_y_v"][event_idx][candidate_muon_index],
            data["trk_sce_start_z_v"][event_idx][candidate_muon_index],
        ]
    )
    muon_end_vector = np.array(
        [
            data["trk_sce_end_x_v"][event_idx][candidate_muon_index],
            data["trk_sce_end_y_v"][event_idx][candidate_muon_index],
            data["trk_sce_end_z_v"][event_idx][candidate_muon_index],
        ]
    )
    # Require the selected muon track to start and end inside the fiducial volume.
    if not in_fv(*muon_start_vector) or not in_fv(*muon_end_vector):
        return False, None

    proton_start_vector = np.array(
        [
            data["trk_sce_start_x_v"][event_idx][candidate_proton_index],
            data["trk_sce_start_y_v"][event_idx][candidate_proton_index],
            data["trk_sce_start_z_v"][event_idx][candidate_proton_index],
        ]
    )
    proton_end_vector = np.array(
        [
            data["trk_sce_end_x_v"][event_idx][candidate_proton_index],
            data["trk_sce_end_y_v"][event_idx][candidate_proton_index],
            data["trk_sce_end_z_v"][event_idx][candidate_proton_index],
        ]
    )
    # Same containment requirement for the proton candidate.
    if not in_fv(*proton_start_vector) or not in_fv(*proton_end_vector):
        return False, None

    # Momentum thresholds: muon uses the range-based muon momentum branch, while
    # proton momentum is computed from the proton kinetic-energy estimate.
    muon_momentum = data["trk_range_muon_mom_v"][event_idx][candidate_muon_index]
    proton_ke_gev = data["trk_energy_proton_v"][event_idx][candidate_proton_index]
    proton_e_gev = proton_ke_gev + PROTON_MASS
    proton_momentum = np.sqrt(proton_e_gev**2 - PROTON_MASS**2)

    if muon_momentum < 0.1 or proton_momentum < 0.25:
        return False, None

    # Muon quality cut: range and MCS momentum estimates should agree to 25%.
    mcs_muon_momentum = data["trk_mcs_muon_mom_v"][event_idx][candidate_muon_index]
    reso = np.abs(muon_momentum - mcs_muon_momentum) / muon_momentum
    if reso > 0.25:
        return False, None

    # Reject likely flipped tracks: the start point should be closer to the
    # reconstructed vertex than the end point for both candidates.
    mu_start_vertex_distance = np.linalg.norm(vertex_vector - muon_start_vector)
    mu_end_vertex_distance = np.linalg.norm(vertex_vector - muon_end_vector)
    proton_start_vertex_distance = np.linalg.norm(vertex_vector - proton_start_vector)
    proton_end_vertex_distance = np.linalg.norm(vertex_vector - proton_end_vector)

    if (
        mu_start_vertex_distance > mu_end_vertex_distance
        or proton_start_vertex_distance > proton_end_vertex_distance
    ):
        return False, None

    # A final topology sanity check: the two starts should be closer to each
    # other than the two ends, consistent with both tracks emerging from the vertex.
    start_to_start_distance = np.linalg.norm(muon_start_vector - proton_start_vector)
    end_to_end_distance = np.linalg.norm(muon_end_vector - proton_end_vector)
    if start_to_start_distance > end_to_end_distance:
        return False, None

    # Proton LLR PID cut. If this passes, the event passes the reco 1mu1p selection.
    if candidate_proton_pid_score >= PROTON_LLR_PID_SCORE:
        return False, None

    muon_theta = data["trk_theta_v"][event_idx][candidate_muon_index]
    muon_phi = data["trk_phi_v"][event_idx][candidate_muon_index]
    proton_theta = data["trk_theta_v"][event_idx][candidate_proton_index]
    proton_phi = data["trk_phi_v"][event_idx][candidate_proton_index]

    candidate_muon_vector = spherical_to_cartesian(muon_momentum, muon_theta, muon_phi)
    candidate_proton_vector = spherical_to_cartesian(proton_momentum, proton_theta, proton_phi)
    muon_e_gev = np.sqrt(muon_momentum**2 + STVTools.MUON_MASS**2)

    stv_tool = STVTools(
        candidate_muon_vector,
        candidate_proton_vector,
        muon_e_gev,
        proton_e_gev,
    )
    return True, stv_values(stv_tool)


def reco_selection(selection_tree, eval_tree, bdt_tree, pfeval_tree, kine_tree, max_events=None):
    n_events = selection_tree.num_entries
    n_to_eval = n_events if max_events is None else min(max_events, n_events)
    selected = np.zeros(n_events, dtype=np.bool_)
    stv_arrays = {
        branch_name: np.full(n_events, INVALID_STV_VALUE, dtype=np.float32)
        for branch_name in STV_BRANCH_NAMES.values()
    }

    # Load the needed branches once, then do the event-by-event logic in Python.
    # This mirrors the notebook while avoiding repeated ROOT branch reads.
    data = load_reco_inputs(
        selection_tree,
        eval_tree,
        bdt_tree,
        pfeval_tree,
        kine_tree,
        entry_stop=n_to_eval,
    )
    print(f"  Evaluating {n_to_eval}/{n_events} events...", flush=True)

    for event_idx in range(n_to_eval):
        passes_selection, event_stv_values = evaluate_reco_event(event_idx, data)
        selected[event_idx] = passes_selection
        if passes_selection:
            for branch_name, value in event_stv_values.items():
                stv_arrays[branch_name][event_idx] = value
        if (event_idx + 1) % 5000 == 0:
            print(f"  Evaluated {event_idx + 1}/{n_to_eval}", flush=True)

    return selected, stv_arrays


def require_matching_entries(tree_map):
    counts = {name: tree.num_entries for name, tree in tree_map.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise RuntimeError(f"Input trees have mismatched entry counts: {counts}")
    return unique_counts.pop()


def write_output_tree(dst, tree_name, selected, stv_arrays, true_1mu1p, truth_category, args):
    output_arrays = {
        args.selected_branch: selected.astype(np.bool_),
    }

    for branch_name, values in stv_arrays.items():
        output_arrays[branch_name] = values.astype(np.float32)

    if not args.skip_truth_categories:
        output_arrays[args.truth_branch] = true_1mu1p.astype(np.bool_)
        output_arrays[args.truth_category_branch] = truth_category.astype(np.int32)

    print(f"Writing '{tree_name}' tree with {len(selected)} entries...")
    with uproot.update(dst) as output_file:
        branch_types = {
            branch_name: values.dtype
            for branch_name, values in output_arrays.items()
        }
        output_tree = output_file.mktree(tree_name, branch_types)
        output_tree.extend(output_arrays)


def main():
    args = parse_args()

    src = args.src
    if args.dst is None:
        stem, ext = os.path.splitext(src)
        dst = stem + "_1mu1p" + ext
    else:
        dst = args.dst

    output_tree_name = args.output_tree

    if os.path.exists(dst):
        print(f"Using existing output copy {dst}.")
        print(f"Existing '{output_tree_name}' tree in that copy will be overwritten.")
    else:
        print(f"Copying {src} -> {dst}...")
        shutil.copy2(src, dst)
        print("Copy complete.")

    print("Reading data from copied file...")
    with uproot.open(dst) as root_file:
        selection_tree = root_file[args.selection_tree]
        eval_tree = root_file[args.eval_tree]
        bdt_tree = root_file[args.bdt_tree]
        pfeval_tree = root_file[args.pfeval_tree]
        kine_tree = root_file[args.kine_tree]

        n_events = require_matching_entries(
            {
                "selection_tree": selection_tree,
                "eval_tree": eval_tree,
                "bdt_tree": bdt_tree,
                "pfeval_tree": pfeval_tree,
                "kine_tree": kine_tree,
            }
        )
        print(f"  {n_events} events loaded.")

        print("Computing reco 1mu1p selection...")
        selected_1mu1p, stv_arrays = reco_selection(
            selection_tree,
            eval_tree,
            bdt_tree,
            pfeval_tree,
            kine_tree,
            max_events=args.max_events,
        )

        if args.skip_truth_categories:
            print("Skipping truth categories.")
            true_1mu1p = None
            truth_category = None
        else:
            print("Computing truth categories...")
            true_1mu1p, truth_category = truth_categories(
                selection_tree,
                eval_tree,
                n_events,
                entry_stop=args.max_events,
            )

    write_output_tree(
        dst,
        output_tree_name,
        selected_1mu1p,
        stv_arrays,
        true_1mu1p,
        truth_category,
        args,
    )

    selected_count = int(np.sum(selected_1mu1p))
    print(f"Selected events: {selected_count}")
    if args.skip_truth_categories:
        print("Truth categories skipped.")
    else:
        signal_count = int(np.sum(true_1mu1p))
        selected_signal_count = int(np.sum(selected_1mu1p & true_1mu1p))
        selected_background_count = int(selected_count - selected_signal_count)

        print("Category codes:")
        for category, name in TRUTH_CATEGORY_NAMES.items():
            print(f"  {int(category)}: {name}")
        print(f"Selected true 1mu1p: {selected_signal_count}")
        print(f"Selected background: {selected_background_count}")
        print(f"All true 1mu1p: {signal_count}")
    print(f"Done. New file: {dst}")


if __name__ == "__main__":
    main()
