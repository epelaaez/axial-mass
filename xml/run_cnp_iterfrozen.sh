#!/usr/bin/env bash
set -uo pipefail

# Full rerun of the analysis with the CNP statistic and the iterated frozen covariance
# (decision of 2026-11-20), into its own output root so the Neyman / dynamic-covariance results stay.
# Needs PROfit develop >= 88c996a (2026-11-23), where the mu of the CNP variance is the full
# prediction, spline shifts included, and the search fix of 2026-11-24 in the runners (--seed and
# --fit-options latin_phys_only 0, see xml/iterate_frozen.sh). Results root axial_mass_cnpmu_iterfrozen
# (first filled 2026-11-22 with a local patch, rerun from scratch with the fixed search; finished fits
# are skipped, so clear everything but the *.bin input caches before a full rerun).
# axial_mass_cnp_iterfrozen (2026-11-20, old CNP with mu = no-spline CV) is superseded.
#
# Steps, in order (each is one of the existing runners with CHI2=CNP COV=iterfrozen):
#   production  xml/run_all.sh, the three families side by side (asimov, nuwro, opendata; 14 fits each)
#   injection   xml/injection_study/run_all.sh       knob ranking: which splines bias F_A when left out
#   ladder      xml/nuisance_ladder_study/run_all.sh  n5..n10: is the result stable when more are floated
#   binning     xml/binning_study/run_all.sh          (nuwro, asimov)
#   lowpn       xml/lowpn_study/run_all.sh
#   oned        xml/oned_study/run_all.sh
#   toys        xml/pseudo_experiments/run_toys.sh    all ensembles, iterfrozen, TOY_SEEDS
# injection and ladder come right after production: together they answer how many splines to float.
#
# A failing step does not stop the following ones; the summary at the end lists what failed, and
# every runner skips nothing on a rerun except the covariance-reference iterations that already
# converged (xml/iterate_frozen.sh), so rerun single steps with --only.
#
# Usage: ./run_cnp_iterfrozen.sh [--only STEP]... [--from STEP] [--list]
# Environment:
#   OUTPUT_ROOT=/nevis/hopper/data/epelaez/axial_mass_cnpmu_iterfrozen
#   CHI2=CNP  COV=iterfrozen  MCMC_ITERATIONS=500000  TOL=0.01  MAX_ITER=30  PROFIT_SEED=1
#   JOBS=4 NTHREADS=8       study runners: JOBS fits at a time, NTHREADS threads each
#   PRODUCTION_NTHREADS=10  threads per production family (three families run at the same time)
#   TOY_SEEDS=1-200 TOY_JOBS=16 TOY_NTHREADS=2 TOY_MCMC_ITERATIONS=20000   the toys step
#   DRY_RUN=1               print the PROfit commands instead of running them
# Logs: OUTPUT_ROOT/logs/<timestamp>_<step>.log

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass_cnpmu_iterfrozen}"
export CHI2="${CHI2:-CNP}"
export COV="${COV:-iterfrozen}"
export MCMC_ITERATIONS="${MCMC_ITERATIONS:-500000}"
export TOL="${TOL:-0.01}"
export MAX_ITER="${MAX_ITER:-30}"
export JOBS="${JOBS:-4}"
export NTHREADS="${NTHREADS:-8}"
export DRY_RUN="${DRY_RUN:-0}"
export PLAIN_DIRS=1          # study runners: plain <family> directories below the new root
production_nthreads="${PRODUCTION_NTHREADS:-10}"

all_steps=(production injection ladder binning lowpn oned toys)
steps=()
from=""
while (($#)); do
    case "$1" in
        --only) (($# >= 2)) || { echo "--only requires a step" >&2; exit 2; }; steps+=("$2"); shift ;;
        --from) (($# >= 2)) || { echo "--from requires a step" >&2; exit 2; }; from="$2"; shift ;;
        --list) printf '%s\n' "${all_steps[@]}"; exit 0 ;;
        -h|--help) sed -n '4,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
if [[ -n "${from}" ]]; then
    keep=0
    for step in "${all_steps[@]}"; do
        [[ "${step}" == "${from}" ]] && keep=1
        [[ "${keep}" == "1" ]] && steps+=("${step}")
    done
    ((${#steps[@]})) || { echo "No such step: ${from} (${all_steps[*]})" >&2; exit 2; }
fi
((${#steps[@]})) || steps=("${all_steps[@]}")
for step in "${steps[@]}"; do
    [[ " ${all_steps[*]} " == *" ${step} "* ]] || { echo "No such step: ${step} (${all_steps[*]})" >&2; exit 2; }
done

# Ctrl-C / kill: background jobs of a script ignore SIGINT, so without this the production families
# (and their PROfit processes) would keep running after the script is interrupted.
trap 'trap - INT TERM; echo "interrupted: stopping all child processes" >&2; kill 0' INT TERM

# One instance per output root: two at once write the same directories.
if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${OUTPUT_ROOT}"
    exec 9> "${OUTPUT_ROOT}/.run_cnp_iterfrozen.lock"
    if ! flock -n 9; then
        echo "Another run_cnp_iterfrozen.sh is already running on ${OUTPUT_ROOT} (lock held). Not starting." >&2
        exit 1
    fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="${OUTPUT_ROOT}/logs"
[[ "${DRY_RUN}" == "1" ]] || mkdir -p "${log_dir}"
failed=()

run_logged() {
    # run_logged <name> <command...>: run with its output in the terminal and in a log; record failures.
    local name="$1"; shift
    local log="${log_dir}/${timestamp}_${name}.log"
    echo "==> [$(date '+%F %T')] ${name}: $*"
    if [[ "${DRY_RUN}" == "1" ]]; then
        "$@" || failed+=("${name}")
        return 0
    fi
    "$@" 2>&1 | tee "${log}"
    if ((PIPESTATUS[0] != 0)); then
        failed+=("${name}")
        echo "!!! [$(date '+%F %T')] ${name} FAILED (log: ${log}); continuing"
    fi
}

production() {
    # The production runner fits one XML at a time, so the three families run side by side.
    local family pids=() status=0
    local -A skip=(
        [asimov]="--no-nuwro --no-opendata" [nuwro]="--no-asimov --no-opendata" [opendata]="--no-asimov --no-nuwro"
    )
    for family in asimov nuwro opendata; do
        if [[ "${DRY_RUN}" == "1" ]]; then
            # shellcheck disable=SC2086
            NTHREADS="${production_nthreads}" "${script_dir}/run_all.sh" ${skip[${family}]} || status=1
            continue
        fi
        # shellcheck disable=SC2086
        NTHREADS="${production_nthreads}" "${script_dir}/run_all.sh" ${skip[${family}]} \
            > "${log_dir}/${timestamp}_production_${family}.log" 2>&1 &
        pids+=("$!")
        echo "    ${family}: pid $!, log ${log_dir}/${timestamp}_production_${family}.log"
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    return "${status}"
}

toys() {
    # One run_toys.sh call for all ensembles: it fits the first toy of each fit alone (input cache).
    JOBS="${TOY_JOBS:-16}" NTHREADS="${TOY_NTHREADS:-2}" MCMC_ITERATIONS="${TOY_MCMC_ITERATIONS:-20000}" \
        "${script_dir}/pseudo_experiments/run_toys.sh" --ensemble A --ensemble B10 --ensemble B11 --ensemble B12 \
        --ensemble C00 --ensemble C12 --ensemble D --method iterfrozen --seeds "${TOY_SEEDS:-1-200}"
}

echo "==> CHI2=${CHI2} COV=${COV} MCMC_ITERATIONS=${MCMC_ITERATIONS} TOL=${TOL} -> ${OUTPUT_ROOT}"
echo "==> steps: ${steps[*]}"
for step in "${steps[@]}"; do
    case "${step}" in
        production) run_logged production production ;;
        injection)  run_logged injection "${script_dir}/injection_study/run_all.sh" ;;
        ladder)     run_logged ladder "${script_dir}/nuisance_ladder_study/run_all.sh" ;;
        binning)    run_logged binning "${script_dir}/binning_study/run_all.sh" ;;
        lowpn)      run_logged lowpn "${script_dir}/lowpn_study/run_all.sh" ;;
        oned)       run_logged oned "${script_dir}/oned_study/run_all.sh" ;;
        toys)       run_logged toys toys ;;
    esac
done

echo "==> [$(date '+%F %T')] finished"
if [[ "${DRY_RUN}" != "1" ]]; then
    # Fits whose profile scan ended away from their covariance reference (xml/iterate_frozen.sh --check).
    mapfile -t mismatches < <(find "${OUTPUT_ROOT}" -name '*reference_mismatch.txt' 2>/dev/null | sort)
    if ((${#mismatches[@]})); then
        echo "Best fit more than TOL from the covariance reference in ${#mismatches[@]} fit(s):"
        printf '  %s\n' "${mismatches[@]}"
    fi
fi
if ((${#failed[@]})); then
    echo "Failed steps: ${failed[*]}" >&2
    exit 1
fi
