"""Selected MC events per analysis bin in each detector-variation sample.

Answers the production team's question of where the DetVar samples are statistically
thin. Reads the minimal_detvar_<name>_df.root trees that the DetVarSection of the
production XMLs (xml/opendata, count50) points to and applies the same selection and
binning as the MCFile branches: afro_1mu1p_sel == 1, reco log10(Q^2) x p_n on the
9 x 8 grid, events outside the grid dropped; signal / background is afro_1mu1p_true.

Each variation only covers some of the CV run periods (the LY ones Run 1, Run 3b and
Run 5; SCE, Recomb2 and WireMod only Run 5), so every variation is compared with the
CV events from the (filetype, detvar_sample, run period) groups it actually has. Per
bin and variation the table gives
    n_var        selected events in the variation sample (raw, unweighted)
    n_cv         selected events in the matching CV subset (raw)
    n_matched    variation events whose run/subrun/event is also selected in the CV
                 subset; this is what PROfit's event-matched DetVar ratio uses
    neff_var     (sum w)^2 / sum w^2 of the variation, the effective count after the
                 per-run-period POT weights
and the same split into signal and background. The CV sample appears as its own row
(all periods).

Writes tables/detvar_bin_occupancy/{detvar_bin_occupancy.csv, run_period_summary.csv}
and figs/detvar_bin_occupancy.pdf:

    python python/scripts/detvar_bin_occupancy.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = Path("/nevis/riverside/data/epelaez/ngem/intermediate_files")
TABLE_DIR = REPO_ROOT / "tables" / "detvar_bin_occupancy"
FIGURE = REPO_ROOT / "figs" / "detvar_bin_occupancy.pdf"

VARIATIONS = ["Recomb2", "LYAtt", "LYDown", "LYRayleigh", "SCE",
              "WireModX", "WireModYZ", "WireModThetaXZ", "WireModThetaYZ"]
# Production reco binning (xml/opendata, count50).
Q2_LOG_EDGES = np.array([-2.00, -1.16, -0.94, -0.80, -0.68, -0.57, -0.47, -0.34, -0.25, 0.20])
PN_EDGES = np.array([0.00, 0.125, 0.175, 0.25, 0.30, 0.40, 0.525, 0.625, 1.00])
# Bins below this many raw variation events are flagged in the figure.
LOW_COUNT = 50

# First run of each MicroBooNE run period.
RUN_PERIOD_STARTS = [(4952, "Run 1"), (8317, "Run 2"), (13697, "Run 3"), (18961, "Run 4a"),
                     (19753, "Run 4b"), (21286, "Run 4c"), (22270, "Run 4d"), (24320, "Run 5")]
KEY = ["detvar_sample", "run", "subrun", "event"]
COLUMNS = KEY + ["filetype", "afro_1mu1p_sel", "afro_1mu1p_true", "afro_1mu1p_Q2",
                 "afro_1mu1p_Pn", "net_weight"]


def run_period(run):
    starts = np.array([s for s, _ in RUN_PERIOD_STARTS])
    labels = np.array([name for _, name in RUN_PERIOD_STARTS])
    return labels[np.searchsorted(starts, run, side="right") - 1]


def load(name):
    arrays = uproot.open(INPUT_DIR / f"minimal_detvar_{name}_df.root")["tree"].arrays(COLUMNS, library="np")
    df = pd.DataFrame({k: v.astype(str) if v.dtype == object else v for k, v in arrays.items()})
    df["period"] = run_period(df["run"].to_numpy())
    df = df[df["afro_1mu1p_sel"] == 1].copy()
    ix = np.digitize(np.log10(df["afro_1mu1p_Q2"]), Q2_LOG_EDGES) - 1
    iy = np.digitize(df["afro_1mu1p_Pn"], PN_EDGES) - 1
    inside = (ix >= 0) & (ix < len(Q2_LOG_EDGES) - 1) & (iy >= 0) & (iy < len(PN_EDGES) - 1)
    df["ix"], df["iy"] = ix, iy
    df["signal"] = df["afro_1mu1p_true"] == 1
    return df[inside]


def per_bin(df, prefix):
    """Raw counts and effective counts per (ix, iy), total and split by signal."""
    grid = pd.MultiIndex.from_product([range(len(Q2_LOG_EDGES) - 1), range(len(PN_EDGES) - 1)],
                                      names=["ix", "iy"])
    out = {}
    for label, sub in [("", df), ("_sig", df[df["signal"]]), ("_bkg", df[~df["signal"]])]:
        g = sub.groupby(["ix", "iy"])["net_weight"]
        out[f"n_{prefix}{label}"] = g.size().reindex(grid, fill_value=0)
        if label == "":
            w2 = g.apply(lambda w: (w ** 2).sum()).reindex(grid, fill_value=0.0)
            sw = g.sum().reindex(grid, fill_value=0.0)
            out[f"neff_{prefix}"] = (sw ** 2 / w2.where(w2 > 0)).fillna(0.0)
            out[f"wsum_{prefix}"] = sw
    return pd.DataFrame(out)


def main():
    cv = load("CV")
    groups = ["filetype", "detvar_sample", "period"]
    rows, period_rows = [], []

    cv_table = per_bin(cv, "var").reset_index()
    cv_table.insert(0, "variation", "CV")
    rows.append(cv_table)
    for (ft, ds, per), sub in cv.groupby(groups):
        period_rows.append(dict(variation="CV", filetype=ft, detvar_sample=ds, period=per, n_selected=len(sub)))

    for name in VARIATIONS:
        var = load(name)
        covered = var[groups].drop_duplicates()
        cv_sub = cv.merge(covered, on=groups)
        matched = var.merge(cv_sub[KEY], on=KEY)
        table = per_bin(var, "var").join(per_bin(cv_sub, "cv")[["n_cv", "n_cv_sig", "n_cv_bkg"]])
        table["n_matched"] = per_bin(matched, "m")["n_m"]
        table = table.reset_index()
        table.insert(0, "variation", name)
        rows.append(table)
        for (ft, ds, per), sub in var.groupby(groups):
            n_cv = len(cv_sub[(cv_sub.filetype == ft) & (cv_sub.detvar_sample == ds) & (cv_sub.period == per)])
            period_rows.append(dict(variation=name, filetype=ft, detvar_sample=ds, period=per,
                                    n_selected=len(sub), n_cv_selected=n_cv))

    table = pd.concat(rows, ignore_index=True)
    table.insert(1, "bin", table["ix"] * (len(PN_EDGES) - 1) + table["iy"])
    table.insert(4, "log10Q2_lo", Q2_LOG_EDGES[table["ix"]])
    table.insert(5, "log10Q2_hi", Q2_LOG_EDGES[table["ix"] + 1])
    table.insert(6, "pn_lo", PN_EDGES[table["iy"]])
    table.insert(7, "pn_hi", PN_EDGES[table["iy"] + 1])
    int_cols = [c for c in table.columns if c.startswith("n_")]
    table[int_cols] = table[int_cols].fillna(-1).astype(int).replace(-1, pd.NA)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "detvar_bin_occupancy.csv", index=False, float_format="%.4g")
    pd.DataFrame(period_rows).to_csv(TABLE_DIR / "run_period_summary.csv", index=False)
    plot(table)
    return table


def plot(table):
    names = ["CV"] + VARIATIONS
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True, constrained_layout=True)
    vmax = table["n_var"].max()
    for ax, name in zip(axes.flat, names):
        sub = table[table["variation"] == name]
        counts = np.zeros((len(Q2_LOG_EDGES) - 1, len(PN_EDGES) - 1))
        counts[sub["ix"], sub["iy"]] = sub["n_var"]
        # Equal-width cells (bin index space) so every count is legible; ticks carry the edges.
        mesh = ax.pcolormesh(counts.T, cmap="BuPu", norm=plt.matplotlib.colors.LogNorm(10, vmax))
        for (i, j), n in np.ndenumerate(counts):
            low = n < LOW_COUNT
            ax.text(i + 0.5, j + 0.5, f"{int(n)}", ha="center", va="center", fontsize=7,
                    color="#b00000" if low else ("white" if n > 0.3 * vmax else "black"),
                    fontweight="bold" if low else "normal")
        ax.set_xticks(range(len(Q2_LOG_EDGES)), [f"{e:g}" for e in Q2_LOG_EDGES], rotation=90, fontsize=7)
        ax.set_yticks(range(len(PN_EDGES)), [f"{e:g}" for e in PN_EDGES], fontsize=7)
        ax.set_title(f"{name}  ({int(sub['n_var'].sum())} selected)", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel(r"reco $\log_{10}(Q^2/\mathrm{GeV}^2)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"reco $p_n$ [GeV/c]")
    fig.colorbar(mesh, ax=axes, label="selected MC events (raw)", shrink=0.8)
    fig.suptitle(f"Selected 1$\\mu$1p events, detector-variation samples "
                 f"(red bold: < {LOW_COUNT})", fontsize=12)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)


if __name__ == "__main__":
    main()
