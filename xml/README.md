# PROfit XML runners

Run all XML measurements below this directory with:

```bash
./run_all.sh
```

The XML configurations are organized as:

```text
  nuwro/
  asimov/
  opendata/
```

PROfit writes the corresponding results below `OUTPUT_ROOT`:

```text
  nuwro_fit_results/<fit>/
  asimov_fit_results/<fit>/
  opendata_fit_results/<fit>/
```

Disable individual families with `--no-nuwro`, `--no-asimov`, or `--no-opendata`. The equivalent environment switches are `RUN_NUWRO=0`, `RUN_ASIMOV=0`, and `RUN_OPENDATA=0`.

To run one fit across the enabled families, provide its XML filename or stem through `FIT`:

```bash
FIT=lqcd_k6 ./run_all.sh
FIT=lqcd_k6.xml ./run_all.sh
```

For multiple fits, give `FIT` a space-separated value or repeat `--fit`:

```bash
FIT="lqcd_k6 minerva_k6" ./run_all.sh
./run_all.sh --fit lqcd_k6 --fit minerva_k6
```

On your machine, set `PROFIT_BIN` to the local PROfit executable:

```bash
PROFIT_BIN=/path/to/PROfit ./run_all.sh
```

Use `./run_all.sh --help` for a concise command reference.

# Finer-binning study

`binning_study/<family>/<variant>/` holds the production XMLs regenerated with
finer reco binning, to test how the measurement improves with more bins. The
families are `nuwro` (NuWro fake data) and `asimov` (GENIE fake data, the
control: any drift of its best fit with the binning is an estimator artifact).
The XMLs are generated from `nuwro/` and `asimov/` and must not be edited by
hand:

```bash
python/scripts/make_binning_study_xmls.py            # write every family and variant
python/scripts/make_binning_study_xmls.py --check    # also print raw MC/NuWro occupancy per bin
python/scripts/make_binning_study_xmls.py --family asimov --fit minerva_k6
```

Rerun the generator whenever the production XMLs change. The variants are
defined in `VARIANTS` inside the generator.

Use `python/notebooks/15_binning_occupancy.ipynb` to inspect the MC occupancy
of the variants or of any candidate edges before adding them to `VARIANTS`.

`python/notebooks/17_count_based_binning.ipynb` grows binnings from scratch with
a count-based greedy rule (split the most populated interval at its median while
every cell keeps at least N_min raw MC events and Q^2 bins stay wider than the
reco resolution) and writes them to `binning_study/count_variants.json`, one
variant per N_min (`count400` ... `count10`). The generator merges that file
into `VARIANTS`, so those rungs are ordinary variants for the runner and for
notebook 16.

Run the study with the dedicated runner, which mirrors `run_all.sh`:

```bash
MCMC_ITERATIONS=500000 binning_study/run_all.sh                   # every family, variant and fit
binning_study/run_all.sh --family asimov --variant fine_both --fit minerva_k6
FAMILIES=nuwro FIT="minerva_k6 lqcd_k6" CHI2=CNP binning_study/run_all.sh
```

Results go to `OUTPUT_ROOT/binning_study_fit_results/<family>/<variant>/<fit>/`.
`--chi2`/`CHI2` selects PROfit's statistic (neyman, pearson, CNP, poisson); a
non-default statistic writes to `<family>_<chi2>/` so it never overwrites the
Neyman results. Unlike `run_all.sh`, the study runner does not pass
`--with-covar` to the `plot` stage by default: the covariance PDFs take several
minutes at a few hundred bins and are not needed to compare fits. Set
`PLOT_WITH_COVAR=1` to write them, or `STAGES=profile` to skip plotting
altogether. 

Fits with different binning are compared through their
`<fit>_v1_global_fit.txt`, `<fit>_v1_PROfile.root` (`global_fit_result`,
`one_sigma_errs`, the MCMC chain) and `<fit>_v1_PROfile_points.txt`, which do
not depend on the binning. `python/notebooks/16_binning_study_comparison.ipynb`
does this for one family (`FAMILY`): chi^2/ndof, posterior widths, the F_A
fractional uncertainty, r_A^2 and the profile scans, each relative to
`nominal`, and closes with an overlay of several families.

# PROfit weight conventions

The spline branches in these configurations use two different conventions:

- `*_UBGenie` and the axial-form-factor PCA branches are absolute cross-section weights. Their `include_only_weights` must include selection and non-GENIE factors, but not another central cross-section weight.
- Flux and SCC spline branches are relative multipliers whose zero-knob value is one. Their `include_only_weights` must include the complete central prediction.

Consequently, prior-fit files with `weight_2 = non_genie_net_weight` and `weight_3 = <prior CV>` use `1,2` for absolute branches and `1,2,3` for relative branches. Files with `weight_2 = net_weight` use `1` for absolute branches and `1,2` for relative branches.

Detector variations remain factorized around the detector-variation sample's own `net_weight`.
