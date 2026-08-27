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

# PROfit weight conventions

The spline branches in these configurations use two different conventions:

- `*_UBGenie` and the axial-form-factor PCA branches are absolute cross-section weights. Their `include_only_weights` must include selection and non-GENIE factors, but not another central cross-section weight.
- Flux and SCC spline branches are relative multipliers whose zero-knob value is one. Their `include_only_weights` must include the complete central prediction.

Consequently, prior-fit files with `weight_2 = non_genie_net_weight` and `weight_3 = <prior CV>` use `1,2` for absolute branches and `1,2,3` for relative branches. Files with `weight_2 = net_weight` use `1` for absolute branches and `1,2` for relative branches.

Detector variations remain factorized around the detector-variation sample's own `net_weight`.
