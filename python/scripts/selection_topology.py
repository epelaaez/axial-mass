"""Truth topology of every selected MC event, at the thresholds of the 1mu1p signal definition.

The per-particle truth arrays (wc_truth_pdg, wc_truth_mother, wc_truth_startMomentum) are
dropped from all_df.parquet, and the scalar truth columns kept there use other thresholds
(35 MeV proton kinetic energy, pre-FSI GENIE pion counts), so the topology is rebuilt from
the raw Wire-Cell ntuples (T_PFeval) for the selected events only. Primary final-state
particles (truth_mother == 0) are counted with the afro_1mu1p_selection thresholds:
muons >= 0.1 GeV/c, protons >= 0.3 GeV/c, charged pions >= 0.07 GeV/c, every pi0 and every
other meson. The signal flag rebuilt here is checked against afro_1mu1p_true.

Output: tables/selection_topology.parquet, one row per selected prediction MC event
(nu, nue, NC pi0, numu CC pi0 and dirt overlays) keyed by (filename, run, subrun, event):
the particle counts, the leading muon and proton momenta and the ``topology`` label used by
5_publication_plots.ipynb. Takes a few minutes (reads ~10 M events from ~30 files).

    python python/scripts/selection_topology.py
"""

from pathlib import Path
import sys
import time

import awkward as ak
import numpy as np
import polars as pl
import uproot

REPO_ROOT = Path(__file__).resolve().parents[2]
NGEM_SRC = REPO_ROOT.parent / "uboone_ngem" / "src"
sys.path.insert(0, str(NGEM_SRC))
from afro_1mu1p_selection import (  # noqa: E402
    BORDERX, BORDERY, BORDERZ, CHARGED_PION_MOMENTUM_THRESHOLD, FVX, FVY, FVZ,
    MUON_MOMENTUM_MAX, MUON_MOMENTUM_THRESHOLD, PROTON_MOMENTUM_MAX, PROTON_MOMENTUM_THRESHOLD,
)
from file_locations import data_files_location, intermediate_files_location  # noqa: E402

ALL_DF = Path(intermediate_files_location) / "all_df.parquet"
RAW_DIR = Path(data_files_location)
OUTPUT = REPO_ROOT / "tables" / "selection_topology.parquet"
PREDICTION_MC_FILETYPES = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "numucc_pi0_overlay",
                           "dirt_overlay"]
KEYS = ["filename", "run", "subrun", "event"]

# Stack categories, in stacking order. Each selected prediction event gets exactly one.
TOPOLOGIES = [
    "CC1mu1p",          # the signal definition
    "CC1muNp",          # >= 2 protons above threshold, no pion or other meson
    "CC1mu0p",          # no proton above threshold, no pion or other meson
    "CC1muNpi",         # at least one charged pion, pi0 or other meson
    "other_numuCC",     # numu CC out of the FV, or 1mu1p with the muon/proton outside the
                        # momentum ranges, or no/several muons above threshold
    "other_nu",         # NC, nue, anti-neutrino interactions
    "dirt",             # dirt overlay
    "cosmic",           # beam-off (EXT), labelled by the notebook, not here
]


def _is_meson(abs_pdg):
    """Vectorised afro_1mu1p_selection.is_meson_or_antimeson."""
    return ((abs_pdg < 9900000) & ((abs_pdg // 1000) % 10 == 0) & ((abs_pdg // 100) % 10 != 0)
            & ~((abs_pdg >= 901) & (abs_pdg <= 930))
            & ~np.isin(abs_pdg, [110, 990, 998, 999, 100]))


def particle_counts(root_path, wanted_keys):
    """Primary-particle counts for the events of one ntuple whose (run, subrun, event)
    are in ``wanted_keys``; returns a polars DataFrame."""
    with uproot.open(root_path) as root_file:
        eval_tree = root_file["wcpselection"]["T_eval"]
        keys = eval_tree.arrays(["run", "subrun", "event"], library="np")
        key_table = pl.DataFrame({k: keys[k].astype(np.int64) for k in ("run", "subrun", "event")})
        key_table = key_table.with_row_index("entry")
        selected = key_table.join(wanted_keys, on=["run", "subrun", "event"], how="inner")
        if selected.is_empty():
            return None
        entries = np.sort(selected["entry"].to_numpy())
        pf_tree = root_file["wcpselection"]["T_PFeval"]
        if pf_tree.num_entries != eval_tree.num_entries:
            raise ValueError(f"{root_path}: T_PFeval and T_eval are not entry-aligned")
        # Read the contiguous span that holds the selected entries, then pick them.
        arrays = pf_tree.arrays(["truth_pdg", "truth_mother", "truth_startMomentum"],
                                entry_start=int(entries[0]), entry_stop=int(entries[-1]) + 1)
        arrays = arrays[entries - entries[0]]

    primary = arrays["truth_mother"] == 0
    pdg = arrays["truth_pdg"]
    abs_pdg = abs(pdg)
    p3 = arrays["truth_startMomentum"][:, :, :3]
    momentum = np.sqrt(ak.sum(p3 ** 2, axis=-1))
    muon = primary & (pdg == 13) & (momentum >= MUON_MOMENTUM_THRESHOLD)
    proton = primary & (pdg == 2212) & (momentum >= PROTON_MOMENTUM_THRESHOLD)
    charged_pion = primary & (abs_pdg == 211) & (momentum >= CHARGED_PION_MOMENTUM_THRESHOLD)
    neutral_pion = primary & (pdg == 111)
    is_meson = ak.Array(_is_meson(ak.to_numpy(ak.flatten(abs_pdg))))
    is_meson = ak.unflatten(is_meson, ak.num(abs_pdg))
    other_meson = primary & (abs_pdg != 211) & is_meson

    def count(mask):
        return ak.to_numpy(ak.sum(mask, axis=1)).astype(np.int64)

    def leading(mask):
        return ak.to_numpy(ak.fill_none(ak.max(ak.mask(momentum, mask), axis=1), np.nan))

    return pl.DataFrame({
        "run": keys["run"][entries].astype(np.int64),
        "subrun": keys["subrun"][entries].astype(np.int64),
        "event": keys["event"][entries].astype(np.int64),
        "n_muon": count(muon), "n_proton": count(proton), "n_charged_pion": count(charged_pion),
        "n_pi0": count(neutral_pion), "n_other_meson": count(other_meson),
        "leading_muon_p": leading(muon), "leading_proton_p": leading(proton),
    })


def selected_events():
    """Keys and scalar truth of every selected prediction MC event in all_df."""
    columns = [*KEYS, "filetype", "afro_1mu1p_true", "wc_truth_nuPdg", "wc_truth_isCC",
               "wc_truth_vtxX", "wc_truth_vtxY", "wc_truth_vtxZ"]
    return (pl.scan_parquet(ALL_DF)
            .filter(pl.col("afro_1mu1p_sel") & pl.col("filetype").is_in(PREDICTION_MC_FILETYPES))
            .select(columns).collect(engine="streaming"))


def label_topology(table):
    in_fv = ((pl.col("wc_truth_vtxX") > BORDERX) & (pl.col("wc_truth_vtxX") < FVX - BORDERX)
             & (pl.col("wc_truth_vtxY") > -FVY / 2 + BORDERY) & (pl.col("wc_truth_vtxY") < FVY / 2 - BORDERY)
             & (pl.col("wc_truth_vtxZ") > BORDERZ) & (pl.col("wc_truth_vtxZ") < FVZ - BORDERZ))
    numu_cc = (pl.col("wc_truth_nuPdg") == 14) & (pl.col("wc_truth_isCC") == 1)
    no_meson = (pl.col("n_charged_pion") == 0) & (pl.col("n_pi0") == 0) & (pl.col("n_other_meson") == 0)
    one_muon = pl.col("n_muon") == 1
    signal = (numu_cc & in_fv & one_muon & (pl.col("n_proton") == 1) & no_meson
              & (pl.col("leading_muon_p") <= MUON_MOMENTUM_MAX)
              & (pl.col("leading_proton_p") <= PROTON_MOMENTUM_MAX))
    topology = (pl.when(pl.col("filetype") == "dirt_overlay").then(pl.lit("dirt"))
                .when(~numu_cc).then(pl.lit("other_nu"))
                .when(signal).then(pl.lit("CC1mu1p"))
                .when(~in_fv | ~one_muon).then(pl.lit("other_numuCC"))
                .when(~no_meson).then(pl.lit("CC1muNpi"))
                .when(pl.col("n_proton") >= 2).then(pl.lit("CC1muNp"))
                .when(pl.col("n_proton") == 0).then(pl.lit("CC1mu0p"))
                .otherwise(pl.lit("other_numuCC")))   # 1mu1p outside the momentum ranges
    return table.with_columns([signal.alias("rebuilt_signal"), topology.alias("topology")])


def main():
    start = time.time()
    selected = selected_events()
    print(f"{len(selected)} selected prediction MC events in all_df ({time.time() - start:.0f}s)")
    pieces = []
    for filename in sorted(selected["filename"].unique().to_list()):
        wanted = selected.filter(pl.col("filename") == filename).select(["run", "subrun", "event"])
        counts = particle_counts(RAW_DIR / filename, wanted)
        if counts is None:
            raise ValueError(f"{filename}: none of its selected events were found in the ntuple")
        pieces.append(counts.with_columns(pl.lit(filename).alias("filename")))
        print(f"  {filename}: {len(counts)}/{len(wanted)} ({time.time() - start:.0f}s)")
    counts = pl.concat(pieces)
    table = label_topology(selected.join(counts, on=KEYS, how="left"))
    missing = table.filter(pl.col("n_muon").is_null())
    if not missing.is_empty():
        raise ValueError(f"{len(missing)} selected events not found in the ntuples")
    agree = (table["rebuilt_signal"] == (table["afro_1mu1p_true"] == 1)).mean()
    print(f"rebuilt signal flag agrees with afro_1mu1p_true for {agree:.5f} of the events")
    if agree < 0.999:
        raise ValueError("rebuilt signal definition disagrees with afro_1mu1p_true")
    OUTPUT.parent.mkdir(exist_ok=True)
    table.write_parquet(OUTPUT)
    print(table["topology"].value_counts().sort("count", descending=True))
    print(f"wrote {OUTPUT} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
