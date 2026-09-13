"""Generate the three deliberately small suite notebooks."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    ("nuwro", "nuwro_fit_results", "NuWro fit results"),
    ("asimov", "asimov_fit_results", "Asimov fit results"),
    ("opendata", "opendata_fit_results", "Open-data fit results"),
)


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


for xml_family, suite, title in SUITES:
    notebook = {
        "cells": [
            cell("markdown", f"# Post-fit pulls in physical parameters — {title}\n\n"
                 "This compact notebook transforms each fit's **joint MCMC chain** from standardized "
                 "fit coordinates to the physical $M_A$ or complete $z$-expansion coefficient vector. "
                 "For correlated priors it applies the inverse PCA map. The MINERvA kmax=6 "
                 "uniform fit uses the same PCA spline directions with flat penalties; its corner "
                 "plot is posterior-only because no informative prior enters "
                 "the fit. Dependent coefficients preserve "
                 "$F_A(0)$ and the four sum rules."),
            cell("code", "from pathlib import Path\nimport sys\n\n"
                 "repo = Path.cwd().resolve()\nwhile repo.name != 'axial_mass' and repo != repo.parent:\n    repo = repo.parent\n"
                 "helper_dir = repo / 'ma_zexp' / 'python' / 'scripts'\nif str(helper_dir) not in sys.path:\n    sys.path.insert(0, str(helper_dir))\n\n"
                 "from postfit_physical_parameters import (\n"
                 "    FIGURE_ROOT, plot_ma_posterior_overlay, run_suite,\n"
                 ")"),
            cell("markdown", "## Run every measurement\n\n"
                 "Set `BURN_IN` or `THIN` if needed. Each measurement produces one table and one "
                 "two-row figure: transformed marginal distributions above, prior/post-fit intervals below."),
            cell("code", "# Optional: fix the display range for parameters shared across fits (e.g. all\n"
                 "# M_A fits, or a1/a2 across every z-expansion prior) so their marginal and\n"
                 "# corner panels are directly comparable. Keys are parameter names as they\n"
                 "# appear in a fit's results table (e.g. 'M_A [GeV]', 'a1', 'a2', ...);\n"
                 "# parameters not listed here keep their automatic, per-fit range. Leave this\n"
                 "# dict empty to keep every parameter's range automatic, as before.\n\n"
                 "# Names:\n"
                 "# M_A [GeV]\n"
                 "# AxFFCCQEshape pull \n"
                 "# NormCCMEC pull\n"
                 "# RPA CCQE pull\n\n"
                 "AXIS_RANGES = {\n"
                 "    'M_A [GeV]': (0.8, 1.8),\n"
                 "    'NormCCMEC pull': (-3, 3),\n"
                 "    'RPA CCQE pull': (-3, 3),\n"
                 "}"),
            cell("code", f"BURN_IN = 0\nTHIN = 1\nN_PRIOR_SAMPLES = 50_000\n"
                 "SHOW_PRIOR_IN_CORNER = True\nSAVE_FIGURES = True\nSAVE_DPI = 600\n\n"
                 f"results = run_suite(\n    '{suite}', BURN_IN, THIN, N_PRIOR_SAMPLES,\n"
                 "    show_prior_in_corner=SHOW_PRIOR_IN_CORNER,\n"
                 "    save_figures=SAVE_FIGURES, save_dpi=SAVE_DPI,\n"
                 "    axis_ranges=AXIS_RANGES,\n)"),
            cell("markdown", "## $M_A$ fit with and without its pull penalty\n\n"
                 "This corner overlays the joint posteriors for physical $M_A$, `NormCCMEC`, "
                 "and `RPA_CCQE`. Both fits omit `AxFFCCQEshape`; the only prior difference "
                 "is that `ma_uniform` removes the Gaussian pull penalty from $M_A$."),
            cell("code", "MA_OVERLAY_FITS = ('ma_no_axff', 'ma_uniform')\n"
                 "MA_OVERLAY_LABELS = {\n"
                 "    'ma_no_axff': r'Posterior from Gaussian $M_A$ prior',\n"
                 "    'ma_uniform': r'Posterior from uniform $M_A$ prior',\n"
                 "}\n\n"
                 "ma_overlay_corner = plot_ma_posterior_overlay(\n"
                 "    results, MA_OVERLAY_FITS, labels=MA_OVERLAY_LABELS,\n"
                 "    figsize=(7.2, 7.2),\n"
                 ")\n"
                 f"ma_overlay_dir = FIGURE_ROOT / '{suite}' / 'comparison_overlays'\n"
                 "ma_overlay_dir.mkdir(parents=True, exist_ok=True)\n"
                 "for extension in ('pdf',):\n"
                 "    ma_overlay_corner.savefig(\n"
                 "        ma_overlay_dir / f'ma_overlay.{extension}', dpi=600,\n"
                 "        bbox_inches='tight', pad_inches=.03, facecolor='white',\n"
                 "    )\n"
                 "ma_overlay_corner"),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.9"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = ROOT / "xml" / xml_family / "postfit_physical_parameters.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(path)
