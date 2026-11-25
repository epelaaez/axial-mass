#!/usr/bin/env bash
set -euo pipefail

# Iterated frozen-covariance fit (PROfit develop >= 9e9d61c).
#
#   iteration 0   PROfit --frozen-cov global                       M frozen at the start reference
#   iteration k   PROfit --frozen-cov --cov-reference-file <fit>_<label>it{k-1}_global_fit.txt global
#   ... until no parameter moves by more than TOL (units of global_fit.txt: prior sigma for the
#   splines) between two consecutive iterations, then
#   final         PROfit --frozen-cov --cov-reference-file <last iteration> <FINAL_STAGES>
#                 written with --output <label>v1 (chain and profile with M frozen at the fixed point).
#
# Start references (--start, repeatable, run one after the other in the same directory so the
# <fit>_*.bin caches are built once):
#   cv        the CV / prior centres (PROfit's default reference)          label ""
#   dynamic   the best fit of the existing dynamic-covariance fit,
#             <study>_fit_results/<family>/<variant>/<fit>/<fit>_v1_global_fit.txt   label "dyn_"
#
# XML tree: STUDY_DIR/<family>/<variant>/<fit>.xml (default: xml/nuisance_ladder_study).
# Results:  OUTPUT_ROOT/<study>_fit_results/<family>_iterfrozen/<variant>/<fit>/
#   <fit>_<label>it<k>_global_fit.txt, <label>it<k>.log/.out   one per iteration
#   <fit>_<label>trajectory.txt                                iteration, chi2, max |delta|, parameters
#   <fit>_<label>v1_*                                          final stage outputs
# <label> = [<chi2>_][dyn_]: the statistic (CHI2, empty for neyman) and the start reference.
# Finished iterations and final stages are skipped, so the script can be rerun to resume.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
study_dir="${STUDY_DIR:-${script_dir}/nuisance_ladder_study}"
study_name="${STUDY_NAME:-$(basename -- "${study_dir}")}"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass}"
study_output="${output_root}/${study_name}_fit_results"
profit_bin="${PROFIT_BIN:-/nevis/riverside/share/epelaez/PROfit/build/bin/PROfit}"
nthreads="${NTHREADS:-8}"
max_jobs="${JOBS:-4}"
tol="${TOL:-0.01}"
max_iter="${MAX_ITER:-8}"
final_stages="${FINAL_STAGES-profile}"
mcmc_iterations="${MCMC_ITERATIONS:-}"
mcmc_burnin="${MCMC_BURNIN:-}"
dry_run="${DRY_RUN:-0}"
# CHI2: PROfit statistic (neyman, pearson, CNP). Anything but the default neyman is passed as
# --chi2 and prefixes every output of the run (<chi2>_it<k>, <fit>_<chi2>_v1_*), in the SAME
# directory as the neyman run so the .bin caches are shared. With --frozen-cov the statistical
# part of M is frozen at the reference prediction too.
chi2="${CHI2:-neyman}"
chi2_args=()
chi2_label=""
if [[ "${chi2,,}" != "neyman" ]]; then
    chi2_args=(--chi2 "${chi2}")
    chi2_label="${chi2,,}_"
fi
# PROFIT_EXTRA: further global PROfit arguments for every fit of the run (e.g. --lndet).
# LABEL: output prefix replacing the one derived from CHI2;
# give one whenever PROFIT_EXTRA changes the fit, so the outputs do not overwrite the plain ones.
read -r -a extra_args <<< "${PROFIT_EXTRA:-}"
chi2_args+=("${extra_args[@]}")
[[ -n "${LABEL:-}" ]] && chi2_label="${LABEL}_"
if ((${#extra_args[@]})) && [[ -z "${LABEL:-}" ]]; then
    echo "PROFIT_EXTRA needs a LABEL for its outputs." >&2
    exit 2
fi

usage() {
    cat <<'USAGE'
Usage: ./run_iterated.sh --family NAME --variant NAME --fit NAME [options]

Options (all repeatable):
  --family NAME    XML family (asimov, nuwro, opendata)
  --variant NAME   variant directory (n5 ... n10 for the nuisance ladder)
  --fit NAME       fit name (ma, minerva_k6_nuisance, ...)
  --start NAME     cv or dynamic (default: cv); see the header of this script

Environment:
  CHI2=neyman         pearson or CNP: passed as --chi2, outputs prefixed pearson_ / cnp_ in the same directory
  PROFIT_EXTRA, LABEL extra global PROfit arguments and the output prefix to use with them
  TOL=0.01            convergence: largest parameter change between two iterations
  MAX_ITER=8          last allowed iteration index; not converged by then = failure, no final stage
  FINAL_STAGES=profile  stages run at the converged reference ("" = iterations only)
  MCMC_ITERATIONS, MCMC_BURNIN   passed to the final profile stage only
  JOBS=4, NTHREADS=8, OUTPUT_ROOT, PROFIT_BIN, STUDY_DIR, STUDY_NAME, DRY_RUN=1
USAGE
}

families=() variants=() fits=() starts=()
while (($#)); do
    case "$1" in
        --family|--variant|--fit|--start)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --family) families+=("$2") ;;
                --variant) variants+=("$2") ;;
                --fit) fits+=("${2%.xml}") ;;
                --start) starts+=("$2") ;;
            esac
            shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
((${#starts[@]})) || starts=(cv)
if ((${#families[@]} == 0 || ${#variants[@]} == 0 || ${#fits[@]} == 0)); then
    echo "Give at least one --family, --variant and --fit." >&2
    usage >&2
    exit 2
fi
for start in "${starts[@]}"; do
    [[ "${start}" == "cv" || "${start}" == "dynamic" ]] || { echo "Unknown --start ${start} (cv, dynamic)" >&2; exit 2; }
done
if [[ ! -x "${profit_bin}" ]]; then
    echo "PROfit executable is not available: ${profit_bin}" >&2
    [[ "${dry_run}" == "1" ]] || exit 1
fi

# max_abs_delta <previous global_fit.txt> <current global_fit.txt>: largest |change| over the
# parameters, matched by position (plotnames need not be unique). Fails if the lists differ.
max_abs_delta() {
    awk -F' : ' '
        FNR == 1 { file++ ; n = 0 }
        NF < 2 { next }
        { n++; value = $NF; name = $0; sub(/ : [^:]*$/, "", name) }
        file == 1 { names[n] = name; previous[n] = value; count = n }
        file == 2 {
            if (name != names[n]) { print "parameter mismatch: " name " vs " names[n] > "/dev/stderr"; bad = 1 }
            d = value - previous[n]; if (d < 0) d = -d
            if (d > max) max = d
            count2 = n
        }
        END { if (bad || count != count2 || count == 0) exit 1; printf "%.6f\n", max }
    ' "$1" "$2"
}

# trajectory_row <iteration> <log> <global_fit.txt> <max delta or -> : one line of the trajectory table.
trajectory_row() {
    local chi2
    chi2="$(grep -a 'Global Best Fit chi.2:' "$2" | tail -n 1 | sed 's/.*chi^2: *//')"
    printf '%s\t%s\t%s' "$1" "${chi2:-nan}" "$4"
    awk -F' : ' 'NF >= 2 { printf "\t%s", $NF }' "$3"
    printf '\n'
}

run_profit() {
    # run_profit <fit_output> <log stem> <command...>
    local fit_output="$1" stem="$2"
    shift 2
    if [[ "${dry_run}" == "1" ]]; then
        printf '        (cd %q &&' "${fit_output}"
        printf ' %q' "$@"
        printf ')\n'
        return 0
    fi
    (cd "${fit_output}" && "$@" > "${stem}.out" 2>&1)
}

run_start() {
    # All iterations and the final stage of one fit from one start reference.
    local family="$1" variant="$2" fit="$3" start="$4"
    local xml="${study_dir}/${family}/${variant}/${fit}.xml"
    local fit_output="${study_output}/${family}_iterfrozen/${variant}/${fit}"
    local name="${family}/${variant}/${fit} [${start}]"
    local label="${chi2_label}" reference=""
    if [[ "${start}" == "dynamic" ]]; then
        label="${chi2_label}dyn_"
        reference="${study_output}/${family}/${variant}/${fit}/${fit}_v1_global_fit.txt"
        if [[ ! -f "${reference}" ]]; then
            echo "missing dynamic best fit for ${name}: ${reference}" >&2
            return 1
        fi
    fi
    local base=("${profit_bin}" --xml "${xml}" --tag "${fit}" --nthread "${nthreads}" --frozen-cov "${chi2_args[@]}")
    local trajectory="${fit_output}/${fit}_${label}trajectory.txt"
    local k out result delta converged=0 reference_args
    [[ "${dry_run}" == "1" ]] || printf 'iteration\tchi2\tmax_abs_delta\tparameters (order of global_fit.txt)\n' > "${trajectory}"

    for ((k = 0; k <= max_iter; k++)); do
        out="${label}it${k}"
        result="${fit_output}/${fit}_${out}_global_fit.txt"
        reference_args=()
        [[ -n "${reference}" ]] && reference_args=(--cov-reference-file "${reference}")
        if [[ "${dry_run}" == "1" ]]; then
            run_profit "${fit_output}" "${out}" "${base[@]}" --output "${out}" --log "${out}.log" "${reference_args[@]}" global
            if ((k == 1)); then
                echo "        ... repeated with the previous iteration as reference until max |delta| < ${tol} (at most it${max_iter})"
                break
            fi
            reference="${result}"
            continue
        fi
        if [[ ! -s "${result}" ]]; then
            run_profit "${fit_output}" "${out}" "${base[@]}" --output "${out}" --log "${out}.log" "${reference_args[@]}" global \
                || { echo "FAILED ${name} ${out} (see ${fit_output}/${out}.out)" >&2; return 1; }
            [[ -s "${result}" ]] || { echo "FAILED ${name} ${out}: no ${result}" >&2; return 1; }
        fi
        delta="-"
        # The first comparison with a file reference is against the reference itself.
        if [[ -n "${reference}" ]]; then
            delta="$(max_abs_delta "${reference}" "${result}")" \
                || { echo "FAILED ${name} ${out}: cannot compare with ${reference}" >&2; return 1; }
        fi
        trajectory_row "${k}" "${fit_output}/${out}.log" "${result}" "${delta}" >> "${trajectory}"
        echo "  ${name} ${out}: max |delta| = ${delta}"
        reference="${result}"
        if [[ "${delta}" != "-" ]] && awk -v d="${delta}" -v t="${tol}" 'BEGIN { exit !(d < t) }'; then
            converged=1
            break
        fi
    done

    if [[ "${dry_run}" != "1" && "${converged}" != "1" ]]; then
        echo "NOT CONVERGED ${name} after it${max_iter} (see ${trajectory})" >&2
        return 1
    fi

    local stage stage_args sub_args
    for stage in ${final_stages}; do
        # Done = the correlation-matrix PDF exists: it is written together with the _mcmc_chain TTree,
        # only with --mcmc-correlations (PROfit develop >= 9e9d61c), which the notebooks need.
        if [[ "${dry_run}" != "1" && "${stage}" == "profile" && -s "${fit_output}/${fit}_${label}v1_PROfile_postfit_correlation_matrix.pdf" ]]; then
            continue
        fi
        stage_args=()
        sub_args=()
        if [[ "${stage}" == "profile" ]]; then
            sub_args+=(--mcmc-correlations)
            [[ -n "${mcmc_iterations}" ]] && stage_args+=(--fit-options MCMC-Iterations "${mcmc_iterations}")
            [[ -n "${mcmc_burnin}" ]] && stage_args+=(--fit-options MCMC-Burnin "${mcmc_burnin}")
        fi
        if [[ "${dry_run}" == "1" ]]; then
            run_profit "${fit_output}" "${label}${stage}" "${base[@]}" --output "${label}v1" --log "${label}${stage}.log" \
                --cov-reference-file "<converged iteration>" "${stage_args[@]}" "${stage}" "${sub_args[@]}"
            continue
        fi
        run_profit "${fit_output}" "${label}${stage}" "${base[@]}" --output "${label}v1" --log "${label}${stage}.log" \
            --cov-reference-file "${reference}" "${stage_args[@]}" "${stage}" "${sub_args[@]}" \
            || { echo "FAILED ${name} final ${stage} (see ${fit_output}/${label}${stage}.out)" >&2; return 1; }
    done
    [[ "${dry_run}" == "1" ]] || echo "done   ${name}: converged at ${out}"
}

run_fit() {
    # The start references of one fit run in sequence: they share the directory and its .bin caches.
    local family="$1" variant="$2" fit="$3" start status=0
    [[ "${dry_run}" == "1" ]] || mkdir -p "${study_output}/${family}_iterfrozen/${variant}/${fit}"
    for start in "${starts[@]}"; do
        run_start "${family}" "${variant}" "${fit}" "${start}" || status=1
    done
    return "${status}"
}

jobs_list=()
for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
        for fit in "${fits[@]}"; do
            xml="${study_dir}/${family}/${variant}/${fit}.xml"
            [[ -f "${xml}" ]] || { echo "No such XML: ${xml}" >&2; exit 2; }
            jobs_list+=("${family} ${variant} ${fit}")
        done
    done
done

echo "==> ${#jobs_list[@]} fits x starts (${starts[*]}), ${max_jobs} at a time, ${nthreads} threads each, TOL ${tol}, final: ${final_stages:-none} -> ${study_output}/<family>_iterfrozen"
[[ "${dry_run}" == "1" ]] && max_jobs=1

failures="$(mktemp)"
trap 'rm -f "${failures}"' EXIT
for job in "${jobs_list[@]}"; do
    read -r family variant fit <<< "${job}"
    while (($(jobs -rp | wc -l) >= max_jobs)); do
        wait -n || true
    done
    [[ "${dry_run}" == "1" ]] || echo "start  ${family}/${variant}/${fit}"
    { run_fit "${family}" "${variant}" "${fit}" || echo "${family}/${variant}/${fit}" >> "${failures}"; } &
done
wait
mapfile -t failed < "${failures}"
if ((${#failed[@]})); then
    echo "Failed or not converged:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
