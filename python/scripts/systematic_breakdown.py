"""Pre-fit systematic uncertainty of the reco (log10 Q^2, p_n) spectrum, split by source.

Reads the per-source covariances that PROfit's ``plot --with-covar`` writes to the
``Covariance`` directory of a ``*_PROplot.root`` file and groups them by the ``tag`` of the
matching ``<systematic>`` in the fit XML. Used by 5_publication_plots.ipynb and
16_binning_study_comparison.ipynb.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

# XML tag -> breakdown group. A tag missing here raises, so a new systematic is never
# silently left out of the groups.
SYST_GROUPS = {
    "Cross section": ("QE", "MEC", "RES", "COH", "NC", "Hadr.", "Non-RES-nu",
                      "Non-RES-antinu", "Struct.", "FSI", "SCC"),
    "Flux": ("Flux", "piprod"),
    "Hadron reinteraction": ("Reint",),
    "Detector": ("DetVar",),
    "POT & targets": ("norm",),
    "MC statistics": ("other",),
}

CHANNEL_CV = "ErrorBand/Var0/nu_uBooNE_numuCC1p_cv2d"

_SYSTEMATIC_ELEMENT = re.compile(r"<systematic\b([^>]*?)(?:/>|>([^<]*)</systematic>)", re.S)


def xml_systematics(xml_path):
    """name -> (tag, type) for every active <systematic>; commented-out ones are skipped."""
    # PROfit XMLs have several top-level elements, so they are not parseable as one tree.
    text = re.sub(r"<!--.*?-->", "", Path(xml_path).read_text(), flags=re.S)
    systematics = {}
    for attributes, body in _SYSTEMATIC_ELEMENT.findall(text):
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', attributes))
        name = attributes.get("name") or body.strip()
        systematics[name] = dict(tag=attributes["tag"], type=attributes["type"])
    return systematics


def has_source_covariances(plot_file):
    """True if the PROplot file was written with --with-covar by a PROfit that stores the
    collapsed per-source matrices."""
    if not Path(plot_file).is_file():
        return False
    with uproot.open(plot_file) as profit_plot_file:
        return "Covariance/collapsed_total_frac_cov" in profit_plot_file


def load_systematic_breakdown(plot_file, xml_path):
    """Per-source pre-fit covariance projected onto log10 Q^2, p_n and the total rate.

    Every source's collapsed fractional covariance is made absolute with the collapsed CV,
    projected as P C P^T and divided by the projected CV again. PROfit's total covariance
    is the sum of the sources, so the group and total uncertainties add the projected
    covariances. Everything is a fraction of the MC prediction, hence independent of the
    POT scale and of the overlaid data.
    """
    systematics = xml_systematics(xml_path)
    tag_group = {tag: group for group, tags in SYST_GROUPS.items() for tag in tags}
    unknown_tags = {s["tag"] for s in systematics.values()} - set(tag_group)
    if unknown_tags:
        raise KeyError(f"Add these XML tags to SYST_GROUPS: {sorted(unknown_tags)}")

    with uproot.open(plot_file) as profit_plot_file:
        cv_hist = profit_plot_file[CHANNEL_CV]
        q2_edges, pn_edges = (axis.edges() for axis in cv_hist.axes)
        # PROfit flattens the grid with Q^2 as the slow index: flat = q2_bin * n_pn + pn_bin.
        central_value = cv_hist.values().reshape(-1)
        n_q2_bins, n_pn_bins = len(q2_edges) - 1, len(pn_edges) - 1
        projections = {
            "q2": np.kron(np.eye(n_q2_bins), np.ones((1, n_pn_bins))),
            "pn": np.kron(np.ones((1, n_q2_bins)), np.eye(n_pn_bins)),
            "rate": np.ones((1, n_q2_bins * n_pn_bins)),
        }

        def project(fractional_covariance):
            absolute = fractional_covariance * np.outer(central_value, central_value)
            return {key: P @ absolute @ P.T for key, P in projections.items()}

        # Collect the matrices by iterating the directory rather than indexing it by name:
        # uproot parses ":" in a lookup as a path separator, and the POT normalization is
        # stored as "collapsed_nu_uBooNE:0.02_cov", which sends uproot 5.0 into a RecursionError.
        collapsed = {
            key.split(";")[0][len("collapsed_"):-len("_cov")]: hist.values()
            for key, hist in profit_plot_file["Covariance"].items(recursive=False)
            if key.startswith("collapsed_") and key.split(";")[0].endswith("_cov")
        }
        total = project(collapsed.pop("total_frac"))
        stored = set(collapsed)
        if stored != set(systematics):
            raise ValueError(
                f"{plot_file}: PROplot covariances and XML systematics differ; rerun the plot "
                f"stage with --with-covar. Only in PROplot: {sorted(stored - set(systematics))}; "
                f"only in XML: {sorted(set(systematics) - stored)}"
            )
        sources = {
            name: dict(group=tag_group[spec["tag"]], tag=spec["tag"], **project(collapsed[name]))
            for name, spec in systematics.items()
        }

    cv_projected = {key: P @ central_value for key, P in projections.items()}
    for key in projections:
        summed = sum(source[key] for source in sources.values())
        if not np.allclose(np.diag(summed), np.diag(total[key]), rtol=1e-3):
            raise ValueError(f"{plot_file}: sources do not add up to PROfit's total covariance ({key})")
    return dict(q2_edges=q2_edges, pn_edges=pn_edges, central_value=cv_projected,
                sources=sources, total=total)


def fractional_uncertainty(breakdown, covariances, key):
    """Fractional 1 sigma on projection ``key`` of the summed ``covariances``."""
    return np.sqrt(np.diag(sum(c[key] for c in covariances))) / breakdown["central_value"][key]


def group_members(breakdown, group):
    """Covariance dicts of one SYST_GROUPS group, or [total] for group "Total"."""
    if group == "Total":
        return [breakdown["total"]]
    return [s for s in breakdown["sources"].values() if s["group"] == group]


def systematic_table(breakdown):
    """One row per source: fractional uncertainty on the rate and the largest per-bin value."""
    rows = []
    for name, source in breakdown["sources"].items():
        rows.append(dict(
            source=name.removesuffix("_UBGenie"), group=source["group"], tag=source["tag"],
            rate=fractional_uncertainty(breakdown, [source], "rate")[0],
            max_q2_bin=fractional_uncertainty(breakdown, [source], "q2").max(),
            max_pn_bin=fractional_uncertainty(breakdown, [source], "pn").max(),
        ))
    table = pd.DataFrame(rows)
    group_order = {group: index for index, group in enumerate(SYST_GROUPS)}
    return (table.assign(_group=table["group"].map(group_order))
            .sort_values(["_group", "rate"], ascending=[True, False])
            .drop(columns="_group").reset_index(drop=True))


def group_table(breakdown):
    """Total and per-group fractional uncertainty on the rate and the largest per-bin value."""
    rows = []
    for group in ["Total", *SYST_GROUPS]:
        members = group_members(breakdown, group)
        rows.append(dict(
            group=group,
            rate=fractional_uncertainty(breakdown, members, "rate")[0],
            max_q2_bin=fractional_uncertainty(breakdown, members, "q2").max(),
            max_pn_bin=fractional_uncertainty(breakdown, members, "pn").max(),
        ))
    return pd.DataFrame(rows)
