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
the count-based grids in `binning_study/count_variants.json`, merged into
`VARIANTS` inside the generator, plus `legacy`, the hand-drawn 8 x 8 grid that
was the production binning until 2026-10-03. Production now uses the `count50`
grid (9 x 8 = 72 bins), the outcome of this study: the finest grid whose DetVar
covariance stays physical (`count35` and finer carry bins with more than 100 %
detector uncertainty) while keeping at least 50 raw MC events per bin and every
bin wider than the reco resolution; finer grids gained no further precision
beyond the MCMC noise.

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
binning_study/run_all.sh --family asimov --variant count50 --fit minerva_k6
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
`REFERENCE_VARIANT`, and closes with an overlay of several families.

# Injection (bias) study

`injection_study/` tests which modeling knobs are degenerate with the axial
form factor: one knob X is injected into the Asimov fake data, held at its CV
in the fit (`--fix`), and the shift of the fitted F_A parameters is the bias X
would cause. PROfit injects and fixes only `type="spline"` systematics, so each
XML is a binning-study base fit with only X turned from `spline_to_covariance`
into `spline`. Every other modeling knob is a `spline_to_covariance`, including
the five knobs the M_A and `*_nuisance` fits float (also in the baselines), so
only the axial parameters are free; everything else keeps its production
treatment.

The study's outcome (2026-10-03, `count50`) fixed the nuisance set of the
production fits: `RPA_CCQE`, `XSecShape_CCMEC`, `MFP_N`, `DecayAngMEC` and
`FrCEx_N` are the knobs whose 1 sigma mis-modelling biases F_A by more than
0.2 delta_F (0.3 to 0.9 delta_F in the M_A fit, RPA alone in the LQCD fit) and
are floated as `type="spline"` in `ma`, `ma_uniform` and every `*_nuisance`
fit; the two-point model switches are restricted to [0, 1]. `NormCCMEC`
(bias 0.02 delta_F) went back into the covariance. The former `ma_no_axff`
is now `ma`: the dipole fit no longer floats `AxFFCCQEshape`, so the Gaussian
and uniform M_A fits differ only in the M_A prior.

1. `python/notebooks/19_modeling_knob_screen.ipynb` screens every `*_UBGenie` and
   `*_SCC` knob in the variant's binning (per-bin `N(k)/N(0)`, as PROfit builds
   it) and writes the knobs worth injecting, their shifts in sigma, the binning
   variant and the base fits to `injection_study/knobs.json`.
2. `python/scripts/make_injection_study_xmls.py` writes
   `injection_study/<variant>/<fit>/<knob>.xml`, a `baseline.xml` per base fit,
   and the run manifest `injection_study/runs.txt`.
3. `injection_study/run_all.sh` runs the manifest:

```bash
injection_study/run_all.sh
injection_study/run_all.sh --fit lqcd_k6 --knob RPA_CCQE_UBGenie
DRY_RUN=1 injection_study/run_all.sh --knob baseline
```

Each XML is processed once in
`OUTPUT_ROOT/injection_study_fit_results/<variant>/<fit>/<knob>/` and its injections reuse the binaries there, one `global` fit each
(`STAGES`), written with `--output inj<sigma>`. The baselines run `profile`
(`BASELINE_STAGES`) for the F_A uncertainties the shifts are compared with.

# PROfit weight conventions

The spline branches in these configurations use two different conventions:

- `*_UBGenie` and the axial-form-factor PCA branches are absolute cross-section weights. Their `include_only_weights` must include selection and non-GENIE factors, but not another central cross-section weight.
- Flux and SCC spline branches are relative multipliers whose zero-knob value is one. Their `include_only_weights` must include the complete central prediction.

Every MC branch therefore has `weight_2 = non_genie_net_weight` and `weight_3` = the central cross-section weight, and uses `1,2` for absolute branches and `1,2,3` for relative branches. In the prior (z-expansion) fits `weight_3` is the prior's CV weight, e.g. `weight_lqcd_k6_FA`. In the dipole M_A fits it is the GENIE tune factor, `net_weight/(non_genie_net_weight + (non_genie_net_weight == 0))`, so the CV prediction is exactly `net_weight`

Detector variations remain factorized around the detector-variation sample's own `net_weight`.
