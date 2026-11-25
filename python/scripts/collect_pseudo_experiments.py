#!/usr/bin/env python3
"""Collect the pseudo-experiment fits of xml/pseudo_experiments/run_toys.sh into one table.

One row per toy fit (ensemble, method, seed): the thrown truth of every floated parameter, the
best fit and chi2 at the minimum, the posterior median and 68 % interval of every parameter when
the fit wrote a chain, and the normalization of the pseudo-data relative to the CV prediction.

The thrown truth and the best fit are parsed from the PROfit log of each toy; the chain and the
data / CV spectra are read from ``<fit>_<ensemble>_<method>_s<seed>_PROglobal.root`` when they are
there. Missing pieces become NaN rather than errors, so the script also works on a partial run.

Output: ``tables/pseudo_experiments/toys.csv`` (all fits) unless --output is given.

Usage:
    python/scripts/collect_pseudo_experiments.py
    python/scripts/collect_pseudo_experiments.py --fit ma --inspect   # list what one toy's ROOT file holds
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("AXIAL_MASS_OUTPUT_ROOT", "/nevis/hopper/data/epelaez/axial_mass_cnpmu_iterfrozen"))
STUDY_ROOT = OUTPUT_ROOT / "pseudo_experiments"
CHANNEL = "nu_uBooNE_numuCC1p"

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TOY_NAME = re.compile(r"^(?P<fit>.+)_(?P<ensemble>[A-Z]\d*)_(?P<method>[a-z]+)_s(?P<seed>\d+)\.log$")
THROWN_LINE = re.compile(r"Pseudo-experiment thrown spline pulls \(sigma\):\s*(.*)")
THROWN_ITEM = re.compile(rf"([\w:.]+)=({NUMBER})( \(injected\))?")
CHI2 = re.compile(rf"Global Best Fit chi\^2:\s*({NUMBER})")
PARAMETER = re.compile(rf"\|\|\s+([\w:.]+)\s+:\s*({NUMBER})\s*$", re.M)

short = lambda name: name.removesuffix("_UBGenie").removeprefix("weight_spline_")


def parse_log(path):
    text = path.read_text(errors="replace")
    row = {}
    thrown = THROWN_LINE.search(text)
    if thrown:
        for name, value, held in THROWN_ITEM.findall(thrown.group(1)):
            row[f"truth_{short(name)}"] = float(value)
            row[f"held_{short(name)}"] = bool(held)
    chi2 = CHI2.findall(text)
    if chi2:
        row["chi2"] = float(chi2[-1])
        block = text.rsplit("Global Best Fit chi^2:", 1)[-1].split("####", 1)[0]
        # Plotnames in the log (MACCQE, FAzexpMinervaK6PCA1, ...) are matched to the spline names
        # case-insensitively by the notebook; here they are stored as PROfit printed them.
        for name, value in PARAMETER.findall(block):
            row[f"best_{name}"] = float(value)
    row["finished"] = "Total run time" in text
    return row


def read_root(path):
    row = {}
    if not path.is_file():
        return row
    with uproot.open(path) as f:
        names = {key.split(";")[0] for key in f.keys()}
        chains = [name for name in names if name.endswith("_mcmc_chain")]
        if chains:
            arrays = f[chains[0]].arrays(library="np")
            for name, values in arrays.items():
                if values.ndim != 1 or values.dtype.kind not in "fiu":
                    continue
                q16, q50, q84 = np.percentile(values, [16, 50, 84])
                row.update({f"med_{short(name)}": q50, f"q16_{short(name)}": q16, f"q84_{short(name)}": q84})
            row["chain_samples"] = len(next(iter(arrays.values())))
        if f"{CHANNEL}_data2d" in names and f"{CHANNEL}_cv2d" in names:
            data = f[f"{CHANNEL}_data2d"].values().sum()
            cv = f[f"{CHANNEL}_cv2d"].values().sum()
            row.update(data_events=float(data), cv_events=float(cv), data_over_cv=float(data / cv) if cv else np.nan)
    return row


def read_trajectory(log):
    """Iterated frozen covariance: number of refits, last max |delta| and the convergence marker."""
    path = log.with_name(log.stem + "_trajectory.txt")
    if not path.is_file():
        return {}
    steps = pd.read_csv(path, sep="\t")
    return dict(refits=len(steps), last_max_delta=steps["max_abs_delta"].iloc[-1] if len(steps) else np.nan,
                converged=log.with_name(log.stem + ".converged").is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", action="append", help="only this fit directory (repeatable)")
    parser.add_argument("--output", type=Path, default=ROOT / "tables" / "pseudo_experiments" / "toys.csv")
    parser.add_argument("--inspect", action="store_true", help="print the contents of one toy's ROOT file and log, then stop")
    args = parser.parse_args()

    directories = [STUDY_ROOT / fit for fit in args.fit] if args.fit else sorted(p for p in STUDY_ROOT.iterdir() if p.is_dir())
    if args.inspect:
        for directory in directories:
            logs = sorted(directory.glob("*_s*.log"))
            if not logs:
                continue
            log = logs[0]
            root_file = log.with_name(log.stem + "_PROglobal.root")
            print(f"== {log}")
            print({key: value for key, value in parse_log(log).items()})
            print(f"== files of this toy: {sorted(p.name for p in directory.glob(log.stem + '*'))}")
            if root_file.is_file():
                with uproot.open(root_file) as f:
                    for key, classname in f.classnames().items():
                        if "slice" not in key:
                            print(f"   {key:60s} {classname}")
        return

    rows = []
    for directory in directories:
        for log in sorted(directory.glob("*_s*.log")):
            match = TOY_NAME.match(log.name)
            if not match:
                continue
            row = dict(match.groupdict())
            row["seed"] = int(row["seed"])
            row.update(parse_log(log))
            row.update(read_root(log.with_name(log.stem + "_PROglobal.root")))
            row.update(read_trajectory(log))
            rows.append(row)
    if not rows:
        raise SystemExit(f"no toy logs below {STUDY_ROOT}")
    table = pd.DataFrame(rows).sort_values(["fit", "ensemble", "method", "seed"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6g")
    summary = table.groupby(["fit", "ensemble", "method"]).agg(toys=("seed", "size"), finished=("finished", "sum"))
    print(summary.to_string())
    print(f"wrote {args.output} ({len(table)} rows, {table.shape[1]} columns)")


if __name__ == "__main__":
    main()
