"""Make the paper-summary F_A figure from the PROfit posterior chains."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from postfit_physical_parameters import (
    FIGURE_ROOT, SPECS, load_fit, plot_fa_summary,
)


# These are the two main figure switches. Use fit keys listed by --list.
DEFAULT_SHOW = ("minerva_k6", "lqcd_k6", "minerva_lqcd_k6")
DEFAULT_COMPARISON_PRIOR = "minerva_k6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="opendata_fit_results")
    parser.add_argument("--show", nargs="*", default=list(DEFAULT_SHOW),
                        help="post-fit distributions to draw (space-separated keys)")
    parser.add_argument("--prior", default=DEFAULT_COMPARISON_PRIOR,
                        help="one fit key whose prior is the comparison band; 'none' disables")
    parser.add_argument("--output", type=Path,
                        help="output stem (default: figs/<suite>/fa_postfit_summary)")
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--q2-max", type=float, default=2.0)
    parser.add_argument("--list", action="store_true", help="list fit keys and exit")
    args = parser.parse_args()

    if args.list:
        for spec in SPECS:
            print(f"{spec.key:24s} {spec.title}")
        return

    requested = set(args.show)
    prior = None if args.prior.lower() == "none" else args.prior
    requested.update(() if prior is None else (prior,))
    specs = {spec.key: spec for spec in SPECS}
    unknown = requested - set(specs)
    if unknown:
        parser.error(f"unknown fit key(s): {', '.join(sorted(unknown))}; use --list")

    results = {}
    for key in requested:
        result = load_fit(specs[key], args.suite, args.burn_in, args.thin)
        if result is None:
            parser.error(f"no unique PROfile ROOT file found for {key!r} in {args.suite!r}")
        results[key] = result

    fig = plot_fa_summary(
        results, show=args.show, comparison_prior=prior,
        q2_range=(0, args.q2_max),
    )
    output = args.output or FIGURE_ROOT / args.suite / "fa_postfit_summary"
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output.with_suffix(f".{extension}")
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=.03,
                    facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
