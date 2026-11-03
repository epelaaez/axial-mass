#!/usr/bin/env bash
set -euo pipefail

# Rerun selected open-data fits with a non-default chi2 statistic and compare the
# post-fit normalization with the Neyman production fits. The Neyman chi2 (stat
# cov = diag(data)) biases the fitted prediction low by ~1 event per bin, ~7% at
# the ~14 events/bin of open data; CNP cancels that bias to first order. Results go
# to OUTPUT_ROOT/chi2_check/<family>_<chi2>/<fit>/, never to the production tree.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass}"
profit_bin="${PROFIT_BIN:-/nevis/riverside/share/epelaez/PROfit/build/bin/PROfit}"
python_bin="${PYTHON_BIN:-/nevis/riverside/share/epelaez/axial_mass/.venv/bin/python}"
nthreads="${NTHREADS:-8}"
family="${FAMILY:-opendata}"
chi2="${CHI2:-CNP}"
mcmc_iterations="${MCMC_ITERATIONS:-500000}"
read -r -a fits <<< "${FITS:-minerva_lqcd_k6 ma}"

if [[ ! -x "${profit_bin}" ]]; then
    echo "PROfit executable is not available: ${profit_bin}" >&2
    exit 1
fi

check_output="${output_root}/chi2_check/${family}_${chi2,,}"

for fit in "${fits[@]}"; do
    xml="${script_dir}/${family}/${fit}.xml"
    if [[ ! -f "${xml}" ]]; then
        echo "No such XML: ${xml}" >&2
        exit 2
    fi
    fit_output="${check_output}/${fit}"
    mkdir -p "${fit_output}"
    echo "==> ${family}/${fit} (--chi2 ${chi2})"
    (
        cd "${fit_output}"
        echo "    plot"
        "${profit_bin}" --xml "${xml}" --tag "${fit}" --output v1 --nthread "${nthreads}" \
            --log plot.log --progress --chi2 "${chi2}" plot --with-splines --with-covar
        echo "    profile"
        "${profit_bin}" --xml "${xml}" --tag "${fit}" --output v1 --nthread "${nthreads}" \
            --log profile.log --progress --chi2 "${chi2}" \
            --fit-options MCMC-Iterations "${mcmc_iterations}" profile
    )
done

echo "==> post-fit normalization, Neyman production vs ${chi2}"
"${python_bin}" - "${output_root}" "${family}" "${check_output}" "${chi2}" "${fits[@]}" <<'EOF'
import sys
import uproot

output_root, family, check_output, chi2, *fits = sys.argv[1:]
channel = "nu_uBooNE_numuCC1p"
for fit in fits:
    for label, path in [("neyman", f"{output_root}/{family}_fit_results/{fit}"),
                        (chi2, f"{check_output}/{fit}")]:
        with uproot.open(f"{path}/{fit}_v1_PROfile.root") as f:
            data = f[f"{channel}_data2d"].values().sum()
            prefit = f[f"{channel}_cv2d"].values().sum()
            postfit = f[f"{channel}_bestfit2d"].values().sum()
        print(f"{fit:16s} {label:7s} data {data:8.1f}  pre-fit {prefit:8.1f}  post-fit {postfit:8.1f}")
EOF
