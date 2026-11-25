# Archive

Scripts that are no longer part of the analysis chain, kept for reference. They
were written for one-off checks or for the method comparison that preceded the
CNP + iterated-frozen-covariance choice. Paths inside them (`script_dir/..`,
`ROOT = parents[2]`) assume their original location, given below; move a script
back there before running it.

| Script | Original location | Purpose |
| --- | --- | --- |
| `xml/run_iterated.sh` | `xml/` | Iterated frozen covariance from two start references (CV, dynamic best fit) on the nuisance-ladder XMLs; inputs of `python/notebooks/27_iterated_frozen_covariance.ipynb`. Superseded by `xml/iterate_frozen.sh`. |
| `xml/run_chi2_check.sh` | `xml/` | Open-data fits with a non-default statistic, compared with the Neyman production fits (2026-10-29). |
| `xml/pseudo_experiments/diagnose_fits.sh` | `xml/pseudo_experiments/` | Diagnostics of the fits and toys whose iteration did not converge (2026-11-24). |
| `python/scripts/toggle_detvar.py` | `python/scripts/` | Comment the detector-variation blocks of the XMLs in and out (DetVar debugging, 2026-09). |
| `python/scripts/spline_reweighting.py` | `python/scripts/` | Importance reweighting of posterior chains for the spline-factorization check (its notebook was removed). |
| `python/scripts/plot_postfit_fa.py` | `python/scripts/` | Command-line F_A summary figure; replaced by `python/notebooks/4_fa_postfit_summary.ipynb`. |
