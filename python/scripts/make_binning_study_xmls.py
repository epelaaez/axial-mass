#!/usr/bin/env python3
"""Generate the finer-binning study XMLs from the production fit XMLs.

Every XML in ``xml/<family>`` (``nuwro`` fake data, ``asimov`` GENIE fake data) is
copied to ``xml/binning_study/<family>/<variant>/`` with the reconstructed
log10(Q^2) vs. p_n binning (the fitted ``var0``) replaced by the variant's
edges. The true-vs-reco log10(Q^2) binning is updated to the same Q^2 edges so
the response matrix stays aligned with the fitted axis.

Two further changes relative to production, both confined to the study:

* The fitted axial-parameter splines (``weight_spline_FAzexp*`` and
  ``MaCCQE_UBGenie``) get ``restrict="-4, 4"`` (``SPLINE_RESTRICT``) unless they
  already carry a ``restrict``. The 7 knots stay at -3..3; PROfit extrapolates
  the outer cubic segment, and the fit box is widened so the finer binnings,
  whose PCA2 minimum sat at the -3 edge, are not clipped.
* Every finer variant keeps at least ``MIN_MC_EVENTS`` raw MC events in every
  reco bin (``--check`` verifies this against the input file and fails
  otherwise). PROfit's MC-statistics covariance takes the inverse of the
  per-bin MC error, so an empty bin makes the covariance singular and a nearly
  empty one is dominated by MC noise. Hence the sparsely populated edge bins
  (lowest and highest log10(Q^2), highest p_n) are never split, and because the
  production corner bin log10(Q^2) in [-2, -1.5] x p_n in [0, 0.1] holds only
  5 MC events, the finer variants move the first p_n edge from 0.10 to 0.15
  (``STUDY_PN``). The ``nominal`` variant keeps the production edges unchanged,
  since it is the reference, and is the one variant exempt from the threshold.

Use ``python/notebooks/15_binning_occupancy.ipynb`` to look at the occupancy of
any candidate binning before adding it to ``VARIANTS``. Count-based variants grown
by ``python/notebooks/17_count_based_binning.ipynb`` are read from
``xml/binning_study/count_variants.json`` and merged into ``VARIANTS``.

Usage:
    python/scripts/make_binning_study_xmls.py            # write all families and variants
    python/scripts/make_binning_study_xmls.py --check    # also report MC occupancy
    python/scripts/make_binning_study_xmls.py --family asimov --fit minerva_k6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = ROOT / "xml" / "binning_study"
# family -> production XML directory the study copies from
FAMILIES = {
    "nuwro": ROOT / "xml" / "nuwro",
    "asimov": ROOT / "xml" / "asimov",
}

# Fit box for the fitted axial-parameter splines, applied unless the XML already restricts them.
SPLINE_RESTRICT = (-4.0, 4.0)
AXIAL_SPLINE_PATTERN = re.compile(r"weight_spline_FAzexp\w*|MaCCQE_UBGenie")

NOMINAL_Q2 = [-2.00, -1.50, -1.20, -1.00, -0.85, -0.70, -0.55, -0.40, -0.20, 0.20]
NOMINAL_PN = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00]
# Baseline p_n axis of the finer variants: first edge moved 0.10 -> 0.15 (see module docstring).
STUDY_PN = [0.00, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00]

# Minimum raw MC events per reco bin for every variant except ``nominal``.
MIN_MC_EVENTS = 10

RECO_UNIT = "log10(Q^2 / GeV^2);p_n"
RESPONSE_UNIT = "True log10(Q^2 / GeV^2);Reco log10(Q^2 / GeV^2)"

INPUT_FILE = "/nevis/riverside/data/epelaez/ngem/intermediate_files/minimal_withspline_df.root"
MC_SELECTION = "isdata==0 && isext==0 && isdirt==0 && isnuwro==0 && afro_1mu1p_sel==1"


def halve(edges, keep=()):
    """Split every bin in two, except the bins whose index is listed in ``keep``.

    Negative indices count from the last bin, as for Python sequences.
    """
    nbins = len(edges) - 1
    keep = {index % nbins for index in keep}
    out = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        out.append(lo)
        if index not in keep:
            out.append(0.5 * (lo + hi))
    out.append(edges[-1])
    return out


# name -> (log10(Q^2) edges, p_n edges, description)
VARIANTS = {
    "nominal": (
        NOMINAL_Q2,
        NOMINAL_PN,
        "production binning (9 x 8 = 72 bins), rerun here as the in-study reference",
    ),
    "fine_q2": (
        halve(NOMINAL_Q2, keep=(0, -1)),
        STUDY_PN,
        "log10(Q^2) bins halved except the first and last; first p_n edge at 0.15 (16 x 8 = 128 bins)",
    ),
    "fine_pn": (
        NOMINAL_Q2,
        halve(STUDY_PN, keep=(0, 1, -1)),
        "p_n bins halved except [0, 0.15], [0.15, 0.20] and [0.7, 1.0] (9 x 13 = 117 bins)",
    ),
    "fine_both": (
        halve(NOMINAL_Q2, keep=(0, -1)),
        halve(STUDY_PN, keep=(0, 1, -1)),
        "both axes refined as above (16 x 13 = 208 bins)",
    ),
    "fine_both_14": (
        halve(NOMINAL_Q2, keep=(0, -1)),
        [0.00, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.00],
        "fine_both with the [0.15, 0.20] p_n bin split; the finest regular refinement (16 x 14 = 224 bins)",
    ),
}

# Count-based variants written by python/notebooks/17_count_based_binning.ipynb.
COUNT_VARIANTS_JSON = TARGET_DIR / "count_variants.json"


def load_json_variants(path=COUNT_VARIANTS_JSON):
    """Extra variants from a JSON file: {name: {"q2": [...], "pn": [...], "description": "..."}}."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    extra = {}
    for name, entry in payload.items():
        if name.startswith("_"):
            continue
        if name in VARIANTS:
            raise ValueError(f"{path}: variant {name!r} clashes with a hard-coded variant")
        extra[name] = (list(entry["q2"]), list(entry["pn"]), entry.get("description", f"from {path.name}"))
    return extra


VARIANTS.update(load_json_variants())

# Variants allowed to violate MIN_MC_EVENTS (the production reference).
THRESHOLD_EXEMPT = {"nominal"}


def fmt_edge(value):
    text = f"{value:.3f}"
    return text[:-1] if text.endswith("0") else text


def fmt_edges(edges):
    return " ".join(fmt_edge(edge) for edge in edges)


BINS2D_RE = re.compile(r"<bins2D\b.*?/>", re.DOTALL)


def set_edges(block, edgesx, edgesy):
    block, nx = re.subn(r'edgesx="[^"]*"', f'edgesx="{fmt_edges(edgesx)}"', block)
    block, ny = re.subn(r'edgesy="[^"]*"', f'edgesy="{fmt_edges(edgesy)}"', block)
    if nx != 1 or ny != 1:
        raise ValueError(f"bins2D block does not carry exactly one edgesx and one edgesy:\n{block}")
    return block


SYSTEMATIC_RE = re.compile(r"<systematic\b([^>]*)>([^<]*)</systematic>")


def widen_axial_splines(text, source_name):
    """Add restrict=SPLINE_RESTRICT to the fitted axial splines that do not restrict yet."""
    lo, hi = SPLINE_RESTRICT
    count = 0

    def replace(match):
        nonlocal count
        attrs, name = match.group(1), match.group(2).strip()
        if 'type="spline"' not in attrs or not AXIAL_SPLINE_PATTERN.fullmatch(name):
            return match.group(0)
        if "restrict=" in attrs:
            return match.group(0)
        count += 1
        return f'<systematic{attrs} restrict="{lo:g}, {hi:g}">{name}</systematic>'

    # Only touch elements outside <!-- --> comments, so commented-out production lines stay as they are.
    parts = re.split(r"(<!--.*?-->)", text, flags=re.DOTALL)
    parts = [part if part.startswith("<!--") else SYSTEMATIC_RE.sub(replace, part) for part in parts]
    return "".join(parts), count


def rebin_xml(text, variant, q2_edges, pn_edges, source_name, family):
    seen = set()

    def replace(match):
        block = match.group(0)
        unit = re.search(r'unit="([^"]*)"', block)
        unit = unit.group(1) if unit else ""
        if unit == RECO_UNIT:
            seen.add("reco")
            return set_edges(block, q2_edges, pn_edges)
        if unit == RESPONSE_UNIT:
            seen.add("response")
            return set_edges(block, q2_edges, q2_edges)
        raise ValueError(f"unexpected bins2D unit {unit!r} in {source_name}")

    text = BINS2D_RE.sub(replace, text)
    if seen != {"reco", "response"}:
        raise ValueError(f"{source_name}: expected reco and response bins2D blocks, found {sorted(seen)}")

    text, widened = widen_axial_splines(text, source_name)
    nq2, npn = len(q2_edges) - 1, len(pn_edges) - 1
    lo, hi = SPLINE_RESTRICT
    header = (
        f"<!-- Generated by python/scripts/make_binning_study_xmls.py from xml/{family}/{source_name}.\n"
        f"     Binning-study variant '{variant}': {VARIANTS[variant][2]}.\n"
        f"     Reco binning {nq2} log10(Q^2) x {npn} p_n = {nq2 * npn} bins;"
        f" {widened} axial spline(s) widened to restrict=\"{lo:g}, {hi:g}\". Do not edit by hand. -->\n"
    )
    declaration = re.match(r"<\?xml[^>]*\?>\s*\n", text)
    if declaration:
        return text[: declaration.end()] + header + text[declaration.end():]
    return header + text


def load_reco_arrays():
    """Return (log10 Q^2, p_n, mc mask, nuwro mask) for the selected events in INPUT_FILE."""
    import numpy as np
    import uproot

    tree = uproot.open(INPUT_FILE)["tree"]
    arrays = tree.arrays(
        ["isdata", "isext", "isdirt", "isnuwro", "afro_1mu1p_sel", "afro_1mu1p_Q2", "afro_1mu1p_Pn"],
        library="np",
    )
    mc = (
        (arrays["isdata"] == 0) & (arrays["isext"] == 0) & (arrays["isdirt"] == 0)
        & (arrays["isnuwro"] == 0) & (arrays["afro_1mu1p_sel"] == 1)
    )
    nuwro = (arrays["isnuwro"] == 1) & (arrays["afro_1mu1p_sel"] == 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(arrays["afro_1mu1p_Q2"])
    return x, arrays["afro_1mu1p_Pn"], mc, nuwro


def check_occupancy(variants):
    """Print per-variant occupancy; return the names that violate MIN_MC_EVENTS."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - depends on the environment
        print(f"occupancy check skipped: {error}", file=sys.stderr)
        return []
    x, y, mc, nuwro = load_reco_arrays()
    print(f"\nRaw event occupancy per reco bin, identical for every family (MC selection: {MC_SELECTION})")
    print(f"{'variant':10s} {'nbins':>5s} {'MC min':>6s} {'MC 5%':>6s} {'MC med':>6s} "
          f"{'MC<' + str(MIN_MC_EVENTS):>6s} {'MC<20':>5s} {'NuWro min':>9s}")
    failing = []
    for name in variants:
        q2, pn, _ = VARIANTS[name]
        h_mc = np.histogram2d(x[mc], y[mc], bins=[q2, pn])[0]
        h_nw = np.histogram2d(x[nuwro], y[nuwro], bins=[q2, pn])[0]
        below = int((h_mc < MIN_MC_EVENTS).sum())
        flag = ""
        if below:
            if name in THRESHOLD_EXEMPT:
                flag = f"  (production reference; {below} bin(s) below threshold tolerated)"
            else:
                flag = f"  <-- {below} bin(s) below MIN_MC_EVENTS={MIN_MC_EVENTS}"
                failing.append(name)
        print(
            f"{name:10s} {h_mc.size:5d} {h_mc.min():6.0f} {np.percentile(h_mc, 5):6.0f} "
            f"{np.median(h_mc):6.0f} {below:6d} {(h_mc < 20).sum():5d} {h_nw.min():9.0f}{flag}"
        )
    return failing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", action="append", choices=sorted(FAMILIES), help="write only this family (repeatable)")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="write only this variant (repeatable)")
    parser.add_argument("--fit", action="append", help="write only this fit stem, e.g. minerva_k6 (repeatable)")
    parser.add_argument("--check", action="store_true", help="report raw MC/NuWro occupancy per bin for each variant")
    args = parser.parse_args()

    variants = args.variant or list(VARIANTS)
    families = args.family or list(FAMILIES)

    for family in families:
        source_dir = FAMILIES[family]
        sources = sorted(source_dir.glob("*.xml"))
        if args.fit:
            wanted = {Path(fit).stem for fit in args.fit}
            sources = [path for path in sources if path.stem in wanted]
            missing = wanted - {path.stem for path in sources}
            if missing:
                parser.error(f"no such fit in {source_dir}: {', '.join(sorted(missing))}")
        if not sources:
            parser.error(f"no XML files found in {source_dir}")

        print(f"== {family} (from {source_dir.relative_to(ROOT)})")
        for name in variants:
            q2, pn, description = VARIANTS[name]
            target = TARGET_DIR / family / name
            target.mkdir(parents=True, exist_ok=True)
            for source in sources:
                text = rebin_xml(source.read_text(), name, q2, pn, source.name, family)
                (target / source.name).write_text(text)
            print(f"   {name:13s} {len(q2) - 1:2d} x {len(pn) - 1:2d} = {(len(q2) - 1) * (len(pn) - 1):3d} bins, "
                  f"{len(sources)} XMLs -> {target.relative_to(ROOT)}")
    for name in variants:
        q2, pn, _ = VARIANTS[name]
        print(f"{name:13s} log10(Q^2): {fmt_edges(q2)}")
        print(f"{'':13s} p_n:        {fmt_edges(pn)}")

    if args.check:
        failing = check_occupancy(variants)
        if failing:
            print(f"\nVariants below MIN_MC_EVENTS={MIN_MC_EVENTS}: {', '.join(failing)}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
