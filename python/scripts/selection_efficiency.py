"""Efficiency, purity and truth composition of the afro 1mu1p selection.

Reads the full uboone_ngem dataframe (all_df.parquet), not the PROfit input tree: the
minimal tree keeps only events with a Wire-Cell reconstructed neutrino (wc_kine_reco_Enu
> 0) and only the BDT test half, so the true-signal denominator is incomplete there.
The prediction is the same set of MC file types the PROfit input holds (data role and
one-gamma samples excluded), weighted with the open-data net weight.

Signal is afro_1mu1p_true (afro_1mu1p_selection.evaluate_true_1mu1p_kinematics): one
numu CC interaction in the fiducial volume with exactly one primary muon in
[0.1, 1.2] GeV/c, one primary proton in [0.3, 1.0] GeV/c, no charged pion above
0.07 GeV/c, no neutral pion and no heavier meson.

Writes tables/selection_efficiency.csv (overall and per true-bin efficiency on the
production binning) and tables/selection_composition.csv (selected sample split by
truth topology x interaction type). 5_publication_plots.ipynb imports the functions
and draws the figures; the script alone writes the tables. Needs polars, which both
project venvs have:

    python python/scripts/selection_efficiency.py
"""

from pathlib import Path
import sys

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
NGEM_SRC = REPO_ROOT.parent / "uboone_ngem" / "src"
sys.path.insert(0, str(NGEM_SRC))
from afro_1mu1p_selection import BORDERX, BORDERY, BORDERZ, FVX, FVY, FVZ  # noqa: E402

ALL_DF = Path("/nevis/riverside/data/epelaez/ngem/intermediate_files/all_df.parquet")
WEIGHT = "wc_net_weight_open_data"
# File types that make up the PROfit prediction (save_PROfit_rootfiles_zexp.build_minimal_df).
PREDICTION_FILETYPES = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "numucc_pi0_overlay",
                        "dirt_overlay", "ext"]
# Production reco binning (xml/opendata, count50), reused for the true-bin efficiency.
Q2_LOG_EDGES = np.array([-2.00, -1.16, -0.94, -0.80, -0.68, -0.57, -0.47, -0.34, -0.25, 0.20])
PN_EDGES = np.array([0.00, 0.125, 0.175, 0.25, 0.30, 0.40, 0.525, 0.625, 1.00])

SCAT_TYPES = {1: "QE", 10: "MEC", 4: "RES", 3: "DIS"}
TOPOLOGIES = ["Signal (true 1mu1p)", "numu CC in FV, not 1mu1p", "numu CC out of FV",
              "Other neutrino (NC, nue, anti-nu)", "Cosmic (EXT)", "Dirt"]


def load_prediction():
    columns = ["filetype", "afro_1mu1p_true", "afro_1mu1p_sel", "wc_kine_reco_Enu", WEIGHT,
               "wc_truth_nuScatType", "wc_truth_nuPdg", "wc_truth_isCC",
               "wc_truth_vtxX", "wc_truth_vtxY", "wc_truth_vtxZ",
               "afro_1mu1p_true_Q2", "afro_1mu1p_true_Pn"]
    in_fv = ((pl.col("wc_truth_vtxX") > BORDERX) & (pl.col("wc_truth_vtxX") < FVX - BORDERX)
             & (pl.col("wc_truth_vtxY") > -FVY / 2 + BORDERY) & (pl.col("wc_truth_vtxY") < FVY / 2 - BORDERY)
             & (pl.col("wc_truth_vtxZ") > BORDERZ) & (pl.col("wc_truth_vtxZ") < FVZ - BORDERZ))
    numu_cc = (pl.col("wc_truth_nuPdg") == 14) & (pl.col("wc_truth_isCC") == 1)
    scat = pl.col("wc_truth_nuScatType").replace_strict(SCAT_TYPES, default="Other")
    topology = (pl.when(pl.col("filetype") == "ext").then(pl.lit(TOPOLOGIES[4]))
                .when(pl.col("filetype") == "dirt_overlay").then(pl.lit(TOPOLOGIES[5]))
                .when(pl.col("afro_1mu1p_true") == 1).then(pl.lit(TOPOLOGIES[0]))
                .when(numu_cc & in_fv).then(pl.lit(TOPOLOGIES[1]))
                .when(numu_cc).then(pl.lit(TOPOLOGIES[2]))
                .otherwise(pl.lit(TOPOLOGIES[3])))
    return (pl.scan_parquet(ALL_DF).select(columns)
            .filter(pl.col("filetype").is_in(PREDICTION_FILETYPES) & pl.col(WEIGHT).is_not_null())
            .with_columns([scat.alias("interaction"), topology.alias("topology"),
                           (pl.col("wc_kine_reco_Enu") > 0).alias("has_wc_reco"),
                           (pl.col("afro_1mu1p_true") == 1).alias("signal"),
                           pl.col("afro_1mu1p_sel").cast(pl.Boolean).alias("selected"),
                           pl.col(WEIGHT).cast(pl.Float64).alias("w")])
            .collect(engine="streaming"))


def efficiency_rows(signal):
    """Overall, staged, per-interaction and per-true-bin efficiency of the signal."""
    total = signal["w"].sum()

    def row(label, mask):
        sub = signal.filter(mask)
        w_all, w_sel = sub["w"].sum(), sub.filter(pl.col("selected"))["w"].sum()
        return dict(selection=label, signal_fraction=w_all / total, n_signal=len(sub),
                    n_selected=int(sub["selected"].sum()),
                    efficiency=w_sel / w_all if w_all else np.nan)

    rows = [row("all", pl.lit(True))]
    with_reco = signal.filter(pl.col("has_wc_reco"))
    rows.append(dict(selection="Wire-Cell reco stage only (wc_kine_reco_Enu > 0)", signal_fraction=1.0,
                     n_signal=len(signal), n_selected=len(with_reco),
                     efficiency=with_reco["w"].sum() / total))
    for name in SCAT_TYPES.values():
        rows.append(row(f"interaction {name}", pl.col("interaction") == name))
    log_q2 = pl.col("afro_1mu1p_true_Q2").log10()
    for lo, hi in zip(Q2_LOG_EDGES[:-1], Q2_LOG_EDGES[1:]):
        rows.append(row(f"true log10 Q^2 in [{lo:.2f}, {hi:.2f})", (log_q2 >= lo) & (log_q2 < hi)))
    pn = pl.col("afro_1mu1p_true_Pn")
    for lo, hi in zip(PN_EDGES[:-1], PN_EDGES[1:]):
        rows.append(row(f"true p_n in [{lo:.3f}, {hi:.3f})", (pn >= lo) & (pn < hi)))
    rows.append(row("outside the production grid",
                    (log_q2 < Q2_LOG_EDGES[0]) | (log_q2 >= Q2_LOG_EDGES[-1]) | (pn >= PN_EDGES[-1])))
    return pl.DataFrame(rows)


def binned_efficiency(signal, variable, edges):
    """Weighted signal and selected-signal sums per bin of a true variable, for the whole
    signal and for each interaction type: {category: (w_all, w_selected)}.

    ``variable`` is "log10_q2" (log10 of the true Q^2) or "pn" (true p_n, GeV/c).
    """
    column = {"log10_q2": np.log10(signal["afro_1mu1p_true_Q2"].to_numpy()),
              "pn": signal["afro_1mu1p_true_Pn"].to_numpy()}[variable]
    weights, selected = signal["w"].to_numpy(), signal["selected"].to_numpy()
    interaction = signal["interaction"].to_numpy()
    categories = {"all": np.ones(len(signal), dtype=bool)}
    categories.update({name: interaction == name for name in SCAT_TYPES.values()})
    return {
        name: (np.histogram(column[mask], edges, weights=weights[mask])[0],
               np.histogram(column[mask & selected], edges, weights=weights[mask & selected])[0])
        for name, mask in categories.items()
    }


def composition_rows(selected):
    """Selected sample split by truth topology and interaction type, weighted fractions."""
    total = selected["w"].sum()
    by_pair = (selected.group_by(["topology", "interaction"])
               .agg([pl.len().alias("n_events"), (pl.col("w").sum() / total).alias("fraction")]))
    by_topology = (selected.group_by("topology")
                   .agg([pl.len().alias("n_events"), (pl.col("w").sum() / total).alias("fraction")])
                   .with_columns(pl.lit("all").alias("interaction")))
    by_interaction = (selected.group_by("interaction")
                      .agg([pl.len().alias("n_events"), (pl.col("w").sum() / total).alias("fraction"),
                            (pl.col("w").filter(pl.col("signal")).sum() / pl.col("w").sum())
                            .alias("signal_fraction_within")])
                      .with_columns(pl.lit("all").alias("topology")))
    table = pl.concat([by_topology, by_pair, by_interaction], how="diagonal")
    order = {name: index for index, name in enumerate(["all", *TOPOLOGIES])}
    return (table.with_columns(pl.col("topology").replace_strict(order).alias("_order"))
            .with_columns((pl.col("interaction") == "all").alias("_summary"))
            .sort(["_order", "_summary", "fraction"], descending=[False, True, True])
            .drop(["_order", "_summary"])
            .select(["topology", "interaction", "n_events", "fraction", "signal_fraction_within"]))


def main():
    prediction = load_prediction()
    signal = prediction.filter(pl.col("signal"))
    selected = prediction.filter(pl.col("selected"))
    efficiency = efficiency_rows(signal)
    composition = composition_rows(selected)
    tables = REPO_ROOT / "tables"
    efficiency.write_csv(tables / "selection_efficiency.csv", float_precision=5)
    composition.write_csv(tables / "selection_composition.csv", float_precision=5)
    purity = selected.filter(pl.col("signal"))["w"].sum() / selected["w"].sum()
    with pl.Config(tbl_rows=100, tbl_width_chars=140, fmt_str_lengths=50):
        print(f"selected prediction: {selected['w'].sum():.1f} weighted events ({len(selected)} MC events)")
        print(f"efficiency {efficiency['efficiency'][0]:.4f}   purity {purity:.4f}")
        print(efficiency)
        print(composition)


if __name__ == "__main__":
    main()
