#!/usr/bin/env bash
set -euo pipefail

# Pseudo-experiments for the covariance-treatment study: each toy is one PROfit
# --pseudo-experiment (floated splines thrown from their priors unless held by --inject-systs,
# covariance shift scaled by the thrown prediction, Poisson) followed by a global fit, once per
# METHOD on the SAME pseudo-data (same --seed). Analysed by python/scripts/collect_pseudo_experiments.py
# (tables/pseudo_experiments/toys.csv).
#
# Ensembles (fit XML from xml/asimov, i.e. the production configuration):
#   A        ma                            nothing held: M_A and the knobs thrown from their priors
#                                          (calibration of the published posterior)
#   B10/B11/B12  ma_uniform                M_A held at 1.0 / 1.1 / 1.2 GeV, knobs thrown
#                                          (bias against a fixed truth, no M_A prior pull)
#   C00/C12  minerva_k6_uniform_nuisance   (PCA1, PCA2) held at (0, 0) / (0, -1.2), knobs thrown
#                                          (can the z expansion recover an injected F_A with a uniform prior;
#                                          (0, -1.2) is close to the open-data best fit)
#   D        minerva_k6_nuisance           nothing held: PCA1, PCA2 and the knobs thrown from their priors
#
# Methods (PROfit develop >= 9e9d61c), named with --method:
#   rescaled    PROfit default: M rebuilt from the prediction at every tested point
#   frozencov   --frozen-cov, M held at the CV prediction
#   iterfrozen  --frozen-cov iterated: the frozencov fit of the same toy is iteration 0 (run first
#               if it is not there yet), then the fit is repeated with --cov-reference-file = the
#               previous best fit until no parameter moves by more than TOL (at most MAX_ITER
#               refits). Every refit overwrites <fit>_<ensemble>_iterfrozen_s<seed>*, so the files
#               left are those of the last iteration (M frozen at the fixed point within TOL);
#               *_trajectory.txt lists chi2 and max |delta| per iteration and *.converged marks success.
#   lndet       --lndet (not pursued, kept for reference)
#
# Statistic: CHI2 (CNP default; neyman, pearson) goes to every method. Every fit passes
# --fit-options latin_phys_only 0 (see xml/iterate_frozen.sh); the toy's own --seed fixes the swarm.
#
# All toys of one fit share one directory, OUTPUT_ROOT/pseudo_experiments/<fit>/, so the input is
# processed into <fit>_*.bin once (the first toy runs alone for that) and every later toy loads it.
# Finished toys (their *_PROglobal.root exists) are skipped, so the script can be rerun to extend
# or resume. Do not rebuild PROfit in the middle of a campaign.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xml_dir="${script_dir}/../asimov"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass_cnpmu_iterfrozen}"
study_output="${output_root}/pseudo_experiments"
profit_bin="${PROFIT_BIN:-/nevis/riverside/share/epelaez/PROfit/build/bin/PROfit}"
nthreads="${NTHREADS:-2}"
max_jobs="${JOBS:-12}"
mcmc_iterations="${MCMC_ITERATIONS:-20000}"
dry_run="${DRY_RUN:-0}"
keep_pdfs="${KEEP_PDFS:-0}"

declare -A ENSEMBLE_FIT=(
    [A]=ma [B10]=ma_uniform [B11]=ma_uniform [B12]=ma_uniform
    [C00]=minerva_k6_uniform_nuisance [C12]=minerva_k6_uniform_nuisance [D]=minerva_k6_nuisance
)
declare -A ENSEMBLE_INJECT=(
    [A]="" [D]=""
    [B10]="MaCCQE_UBGenie -1" [B11]="MaCCQE_UBGenie 0" [B12]="MaCCQE_UBGenie 1"
    [C00]="weight_spline_FAzexpMinervaK6PCA1 0 weight_spline_FAzexpMinervaK6PCA2 0"
    [C12]="weight_spline_FAzexpMinervaK6PCA1 0 weight_spline_FAzexpMinervaK6PCA2 -1.2"
)
declare -A METHOD_ARGS=( [rescaled]="" [frozencov]="--frozen-cov" [iterfrozen]="--frozen-cov" [lndet]="--lndet" )
tol="${TOL:-0.01}"
max_iter="${MAX_ITER:-30}"
chi2="${CHI2:-CNP}"
metric_args=()
[[ "${chi2,,}" != "neyman" ]] && metric_args=(--chi2 "${chi2}")
frozen_args=(--frozen-cov)

usage() {
    cat <<'USAGE'
Usage: ./run_toys.sh --ensemble NAME [--ensemble NAME ...] [options]

Options:
  --ensemble NAME    A, B10, B11, B12, C00, C12 or D (repeatable; see the header of this script)
  --method NAME      rescaled, frozencov, iterfrozen or lndet (repeatable; default: rescaled
                     frozencov iterfrozen; iterfrozen produces the frozencov fit as well)
  --seeds FIRST-LAST Seeds to run, e.g. 1-200 (default 1-200)
  --smoke            Two seeds only, PDFs kept, timing printed: use it once before a campaign
  -h, --help         Show this help

Environment:
  JOBS=12              toys fitted at the same time
  NTHREADS=2           threads per toy
  MCMC_ITERATIONS=20000  chain length per toy
  KEEP_PDFS=1          keep the per-toy PDFs (deleted by default)
  TOL=0.01, MAX_ITER=30 iterfrozen: convergence threshold (largest parameter change, prior sigma)
                       and the largest number of refits
  CHI2=CNP             statistic for every method (default; neyman, pearson)
  OUTPUT_ROOT, PROFIT_BIN, DRY_RUN=1
USAGE
}

ensembles=()
methods=()
seed_first=1
seed_last=200
smoke=0
while (($#)); do
    case "$1" in
        --ensemble) ensembles+=("$2"); shift ;;
        --method) methods+=("$2"); shift ;;
        --seeds) seed_first="${2%-*}"; seed_last="${2#*-}"; shift ;;
        --smoke) smoke=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
((${#ensembles[@]})) || { usage >&2; exit 2; }
((${#methods[@]})) || methods=(rescaled frozencov iterfrozen)
for ensemble in "${ensembles[@]}"; do
    [[ -n "${ENSEMBLE_FIT[${ensemble}]:-}" ]] || { echo "No such ensemble: ${ensemble}" >&2; exit 2; }
done
for method in "${methods[@]}"; do
    [[ -n "${METHOD_ARGS[${method}]+x}" ]] || { echo "No such method: ${method}" >&2; exit 2; }
done
# iterfrozen runs the frozencov fit of its toy itself (as iteration 0): launching both as separate
# jobs would run that fit twice at the same time, so frozencov is dropped from the job list.
if [[ " ${methods[*]} " == *" iterfrozen "* ]]; then
    kept=()
    for method in "${methods[@]}"; do [[ "${method}" == "frozencov" ]] || kept+=("${method}"); done
    methods=("${kept[@]}")
fi
if [[ "${smoke}" == "1" ]]; then
    seed_last=$((seed_first + 1))
    keep_pdfs=1
fi
[[ -x "${profit_bin}" ]] || { echo "PROfit executable is not available: ${profit_bin}" >&2; exit 1; }

# max_abs_delta <previous global_fit.txt> <current global_fit.txt>: largest |change| over the
# parameters, matched by position (same helper as xml/iterate_frozen.sh).
max_abs_delta() {
    awk -F' : ' '
        FNR == 1 { file++ ; n = 0 }
        NF < 2 { next }
        { n++; value = $NF; name = $0; sub(/ : [^:]*$/, "", name) }
        file == 1 { names[n] = name; previous[n] = value; count = n }
        file == 2 {
            if (name != names[n]) bad = 1
            d = value - previous[n]; if (d < 0) d = -d
            if (d > max) max = d
            count2 = n
        }
        END { if (bad || count != count2 || count == 0) exit 1; printf "%.6f\n", max }
    ' "$1" "$2"
}

run_iterated_toy() {
    # run_iterated_toy <ensemble> <seed>: the iterfrozen method (see the header).
    local ensemble="$1" seed="$2"
    local fit="${ENSEMBLE_FIT[${ensemble}]}"
    local dir="${study_output}/${fit}"
    local name="${fit}_${ensemble}_iterfrozen_s${seed}"
    local first="${dir}/${fit}_${ensemble}_frozencov_s${seed}_global_fit.txt"
    [[ -f "${dir}/${name}.converged" ]] && return 0
    if [[ "${dry_run}" == "1" ]]; then
        run_toy "${ensemble}" frozencov "${seed}"
        echo "    then, until max |delta| < ${tol} (at most ${max_iter} times): the same command with --output ${ensemble}_iterfrozen_s${seed} --cov-reference-file <previous global_fit.txt>"
        return 0
    fi
    [[ -s "${first}" ]] || run_toy "${ensemble}" frozencov "${seed}" || return 1
    [[ -s "${first}" ]] || { echo "FAILED ${name}: no ${first}" >&2; return 1; }
    local inject=()
    read -r -a inject <<< "${ENSEMBLE_INJECT[${ensemble}]}"
    local reference="${dir}/${name}_reference.txt" result="${dir}/${name}_global_fit.txt"
    local k delta start=${SECONDS}
    cp "${first}" "${reference}"
    printf 'iteration\tchi2\tmax_abs_delta\n' > "${dir}/${name}_trajectory.txt"
    for ((k = 1; k <= max_iter; k++)); do
        local cmd=("${profit_bin}" --xml "${xml_dir}/${fit}.xml" --tag "${fit}" --output "${ensemble}_iterfrozen_s${seed}"
                   --nthread "${nthreads}" --seed "${seed}" --log "${name}.log" --pseudo-experiment)
        ((${#inject[@]})) && cmd+=(--inject-systs "${inject[@]}")
        cmd+=("${metric_args[@]}" "${frozen_args[@]}" --cov-reference-file "${reference}" --fit-options latin_phys_only 0 MCMC-Iterations "${mcmc_iterations}" global --mcmc-correlations)
        rm -f "${result}"
        (cd "${dir}" && "${cmd[@]}" > "${name}.out" 2>&1) && [[ -s "${result}" ]] \
            || { echo "FAILED ${name} iteration ${k} (see ${dir}/${name}.out)" >&2; return 1; }
        delta="$(max_abs_delta "${reference}" "${result}")" || { echo "FAILED ${name}: parameter lists differ" >&2; return 1; }
        # chi2 of this refit (minimum at the covariance frozen at the previous best fit)
        printf '%s\t%s\t%s\n' "${k}" "$(sed -n '1s/.*chi2 = *\([^ ]*\).*/\1/p' "${result}")" "${delta}" >> "${dir}/${name}_trajectory.txt"
        if awk -v d="${delta}" -v t="${tol}" 'BEGIN { exit !(d < t) }'; then
            touch "${dir}/${name}.converged"
            [[ "${keep_pdfs}" == "1" ]] || rm -f "${dir}/${name}"*.pdf
            [[ "${smoke}" == "1" ]] && echo "done   ${name}  ${k} refits, $((SECONDS - start)) s"
            return 0
        fi
        cp "${result}" "${reference}"
    done
    [[ "${keep_pdfs}" == "1" ]] || rm -f "${dir}/${name}"*.pdf
    echo "NOT CONVERGED ${name} after ${max_iter} refits (last max |delta| ${delta})" >&2
    return 1
}

run_toy() {
    # run_toy <ensemble> <method> <seed>
    local ensemble="$1" method="$2" seed="$3"
    if [[ "${method}" == "iterfrozen" ]]; then
        run_iterated_toy "${ensemble}" "${seed}"
        return
    fi
    local fit="${ENSEMBLE_FIT[${ensemble}]}"
    local dir="${study_output}/${fit}"
    local out="${ensemble}_${method}_s${seed}"
    local name="${fit}_${out}"
    [[ -f "${dir}/${name}_PROglobal.root" ]] && return 0
    local inject=() extra=()
    read -r -a inject <<< "${ENSEMBLE_INJECT[${ensemble}]}"
    read -r -a extra <<< "${METHOD_ARGS[${method}]}"
    local cmd=("${profit_bin}" --xml "${xml_dir}/${fit}.xml" --tag "${fit}" --output "${out}" --nthread "${nthreads}"
               --seed "${seed}" --log "${name}.log" --pseudo-experiment)
    ((${#inject[@]})) && cmd+=(--inject-systs "${inject[@]}")
    cmd+=("${metric_args[@]}")
    ((${#extra[@]})) && cmd+=("${extra[@]}")
    cmd+=(--fit-options latin_phys_only 0 MCMC-Iterations "${mcmc_iterations}" global --mcmc-correlations)
    if [[ "${dry_run}" == "1" ]]; then
        printf '(cd %q &&' "${dir}"; printf ' %q' "${cmd[@]}"; printf ')\n'
        return 0
    fi
    local start=${SECONDS}
    if (cd "${dir}" && "${cmd[@]}" > "${name}.out" 2>&1); then
        [[ "${keep_pdfs}" == "1" ]] || rm -f "${dir}/${name}"*.pdf
        [[ "${smoke}" == "1" ]] && echo "done   ${name}  $((SECONDS - start)) s"
        return 0
    fi
    echo "FAILED ${name} (see ${dir}/${name}.out)" >&2
    return 1
}

failures="$(mktemp)"
trap 'rm -f "${failures}"' EXIT
for ensemble in "${ensembles[@]}"; do
    fit="${ENSEMBLE_FIT[${ensemble}]}"
    dir="${study_output}/${fit}"
    [[ "${dry_run}" == "1" ]] || mkdir -p "${dir}"
    echo "==> ensemble ${ensemble} (${fit}; held: ${ENSEMBLE_INJECT[${ensemble}]:-nothing}), methods ${methods[*]}, seeds ${seed_first}-${seed_last}, ${max_jobs} jobs -> ${dir}"
    if [[ "${dry_run}" != "1" ]]; then
        git -C "$(dirname -- "$(dirname -- "$(dirname -- "${profit_bin}")")")" log -1 --format='PROfit %H %cd' > "${dir}/profit_version_${ensemble}.txt" 2>/dev/null || true
        git -C "$(dirname -- "$(dirname -- "$(dirname -- "${profit_bin}")")")" status --short >> "${dir}/profit_version_${ensemble}.txt" 2>/dev/null || true
    fi
    for seed in $(seq "${seed_first}" "${seed_last}"); do
        for method in "${methods[@]}"; do
            # The first toy of a fit runs alone so that it alone writes the shared *.bin caches.
            if [[ "${dry_run}" == "1" || ! -f "${dir}/${fit}_prop.bin" ]]; then
                wait
                run_toy "${ensemble}" "${method}" "${seed}" || echo "${ensemble} ${method} ${seed}" >> "${failures}"
                continue
            fi
            while (($(jobs -rp | wc -l) >= max_jobs)); do wait -n || true; done
            { run_toy "${ensemble}" "${method}" "${seed}" || echo "${ensemble} ${method} ${seed}" >> "${failures}"; } &
        done
    done
    wait
    if [[ "${dry_run}" != "1" ]]; then
        done_count=$(find "${dir}" -maxdepth 1 -name "${fit}_${ensemble}_*_PROglobal.root" | wc -l)
        echo "    ${done_count} finished toy fits for ensemble ${ensemble}"
    fi
done
if [[ -s "${failures}" ]]; then
    echo "Failed toys (ensemble method seed):" >&2
    sed 's/^/  /' "${failures}" >&2
    exit 1
fi
