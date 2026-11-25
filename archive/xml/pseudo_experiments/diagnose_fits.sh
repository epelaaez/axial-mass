#!/usr/bin/env bash
set -uo pipefail

# Diagnostic runs for the fits whose iterated frozen covariance does not converge (2026-11-24).
#
# Finding from the existing logs: every unconverged fit has two local minima within 0.005-0.6 in chi2
# at its frozen covariance, and PROfit's grad-* presets search only two distinct starting points in an
# all-spline model (latin_phys_only=1 puts all 1000 hypercube points at the origin, so the 16 swarm
# particles and 16 local fits start from the same point; the one-iteration swarm step is the only
# other start). Which minimum is reported then depends on the reference, and the iteration jumps.
# --fit-options latin_phys_only 0 spreads the hypercube over every floated parameter, so the 16 local
# fits start from 16 different points.
#
# Modes (outputs next to the existing files, under new names; nothing existing is touched):
#   stuck    the 10 toys that did not converge on 2026-11-23: iterate from the CV-frozen fit with the
#            full search, NO damping, every iteration kept as <fit>_<ens>_diag_s<seed>_it<k>_global_fit.txt
#            (+ .log); <fit>_<ens>_diag_s<seed>_trajectory.txt lists max |delta| and the local minima found.
#   recheck  converged toys (seeds RECHECK_SEEDS of every ensemble): one fit with the full search at the
#            converged toy's own reference; <fit>_<ens>_recheck_s<seed>_global_fit.txt. Tests whether the
#            two-start search missed a lower minimum in fits that did converge.
#   lowpn    the unconverged low-p_n fit (nuwro/highpn/minerva_k6_uniform_nuisance): xml/iterate_frozen.sh
#            with prefix diag_, the full search and a fixed --seed (the production runners use none, so
#            every iteration's swarm step is different), no damping.
#
#   cycle    the 7 toys of the 2026-11-24 rerun that end in a two-state cycle (CYCLE below; all jump in
#            FrCEx_N across +1): CYCLE_ITER more refits from the toy's current reference, every one kept as
#            <fit>_<ens>_cycle_s<seed>_it<k>_global_fit.txt (+ .log); <fit>_<ens>_cycle_s<seed>_trajectory.txt
#            lists chi2, max |delta|, FrCEx_N and the local minima, i.e. what a delta-chi2 convergence test
#            would see. Then one profile stage at each of the last two references (outputs
#            <fit>_<ens>_cycle_s<seed>_prof<k>_PROfile_points.txt): the chi2 profile along FrCEx_N at both
#            frozen covariances, to see whether the two minima sit on either side of the spline knot at +1.
#            Not part of the default modes.
#
# Usage: ./diagnose_fits.sh [stuck] [recheck] [lowpn] [cycle]     (default: the first three)
# Environment: OUTPUT_ROOT=/nevis/hopper/data/epelaez/axial_mass_cnpmu_iterfrozen, JOBS=10, NTHREADS=2,
#              MAX_ITER=15, RECHECK_SEEDS=1-10, CYCLE_ITER=4, PROFIT_BIN, DRY_RUN=1

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xml_dir="${script_dir}/../asimov"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass_cnpmu_iterfrozen}"
study_output="${output_root}/pseudo_experiments"
profit_bin="${PROFIT_BIN:-/nevis/riverside/share/epelaez/PROfit/build/bin/PROfit}"
nthreads="${NTHREADS:-2}"
max_jobs="${JOBS:-10}"
max_iter="${MAX_ITER:-15}"
tol="${TOL:-0.01}"
recheck_seeds="${RECHECK_SEEDS:-1-10}"
dry_run="${DRY_RUN:-0}"
search=(--fit-options latin_phys_only 0)

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
STUCK=("A 34" "B10 185" "B10 116" "B12 116" "C00 6" "C00 117" "C00 139" "C12 72" "C12 174" "C12 36")
CYCLE=("A 34" "A 38" "B12 46" "B12 175" "C00 6" "C00 116" "C00 132")
cycle_iter="${CYCLE_ITER:-4}"

modes=("$@")
((${#modes[@]})) || modes=(stuck recheck lowpn)
[[ "${dry_run}" == "1" || -x "${profit_bin}" ]] || { echo "PROfit executable is not available: ${profit_bin}" >&2; exit 1; }

max_abs_delta() {  # same helper as xml/iterate_frozen.sh
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
minima() {  # distinct local-minimum chi2 values in a PROfit log, "value xcount" sorted
    grep -a 'Minimization successful, chi' "$1" | grep -ao 'chi [0-9.e+-]*' | awk '{print $2}' | sort -g | uniq -c \
        | awk '{printf "%s%s x%s", (NR > 1 ? " " : ""), $2, $1}'
}

toy_fit() {
    # toy_fit <ensemble> <seed> <output> <reference file or ''>: one global fit of the toy, full search.
    local ens="$1" seed="$2" out="$3" ref="$4" stage="${5:-global}"
    local fit="${ENSEMBLE_FIT[${ens}]}" inject=()
    read -r -a inject <<< "${ENSEMBLE_INJECT[${ens}]}"
    local cmd=("${profit_bin}" --xml "${xml_dir}/${fit}.xml" --tag "${fit}" --output "${out}" --nthread "${nthreads}"
               --seed "${seed}" --log "${fit}_${out}.log" --pseudo-experiment)
    ((${#inject[@]})) && cmd+=(--inject-systs "${inject[@]}")
    cmd+=(--chi2 CNP --frozen-cov)
    [[ -n "${ref}" ]] && cmd+=(--cov-reference-file "${ref}")
    cmd+=("${search[@]}" "${stage}")
    if [[ "${dry_run}" == "1" ]]; then
        printf '(cd %q &&' "${study_output}/${fit}"; printf ' %q' "${cmd[@]}"; printf ')\n'
        return 0
    fi
    (cd "${study_output}/${fit}" && "${cmd[@]}" > "${fit}_${out}.out" 2>&1) \
        && [[ -s "${study_output}/${fit}/${fit}_${out}_global_fit.txt" ]] || return 1
    rm -f "${study_output}/${fit}/${fit}_${out}_PROglobal"* "${study_output}/${fit}/${fit}_${out}"_PROfile*.pdf
}

cycle_toy() {
    local ens="$1" seed="$2"
    local fit="${ENSEMBLE_FIT[${ens}]}" dir="${study_output}/${ENSEMBLE_FIT[${ens}]}"
    local base="${ens}_cycle_s${seed}" trajectory="${dir}/${fit}_${ens}_cycle_s${seed}_trajectory.txt"
    local ref="${dir}/${fit}_${ens}_iterfrozen_s${seed}_reference.txt" k result delta
    [[ -s "${ref}" || "${dry_run}" == "1" ]] || { echo "FAILED ${fit} ${base}: no ${ref}" >&2; return 1; }
    [[ "${dry_run}" == "1" ]] || printf 'iteration\tchi2\tmax_abs_delta\tFrCEx_N\tlocal_minima\n' > "${trajectory}"
    for ((k = 1; k <= cycle_iter; k++)); do
        result="${dir}/${fit}_${base}_it${k}_global_fit.txt"
        if [[ ! -s "${result}" ]]; then
            toy_fit "${ens}" "${seed}" "${base}_it${k}" "${ref}" || { echo "FAILED ${fit} ${base}_it${k}" >&2; return 1; }
        fi
        [[ "${dry_run}" == "1" ]] && { echo "    ... it2..it${cycle_iter} with --cov-reference-file the previous iteration, then profile at the last two references"; return 0; }
        delta="$(max_abs_delta "${ref}" "${result}")"
        printf '%s\t%s\t%s\t%s\t%s\n' "${k}" "$(sed -n '1s/.*chi2 = *\([^ ]*\).*/\1/p' "${result}")" "${delta}" \
            "$(sed -n 's/^FrCEx_N : *//p' "${result}")" "$(minima "${dir}/${fit}_${base}_it${k}.log")" >> "${trajectory}"
        ref="${result}"
    done
    # chi2 profiles at the two states of the cycle: covariance frozen at the last and the last-but-one best fit
    for k in $((cycle_iter - 1)) "${cycle_iter}"; do
        [[ -s "${dir}/${fit}_${base}_prof${k}_PROfile_points.txt" ]] && continue
        toy_fit "${ens}" "${seed}" "${base}_prof${k}" "${dir}/${fit}_${base}_it${k}_global_fit.txt" profile \
            || { echo "FAILED ${fit} ${base}_prof${k}" >&2; return 1; }
    done
    echo "done ${fit} ${base}"
}

stuck_toy() {
    local ens="$1" seed="$2"
    local fit="${ENSEMBLE_FIT[${ens}]}" dir="${study_output}/${ENSEMBLE_FIT[${ens}]}"
    local base="${ens}_diag_s${seed}" trajectory="${dir}/${fit}_${ens}_diag_s${seed}_trajectory.txt"
    local k ref="" result delta
    [[ "${dry_run}" == "1" ]] || printf 'iteration\tmax_abs_delta\tlocal_minima\n' > "${trajectory}"
    for ((k = 0; k <= max_iter; k++)); do
        result="${dir}/${fit}_${base}_it${k}_global_fit.txt"
        if [[ ! -s "${result}" ]]; then
            toy_fit "${ens}" "${seed}" "${base}_it${k}" "${ref}" || { echo "FAILED ${fit} ${base}_it${k}" >&2; return 1; }
        fi
        [[ "${dry_run}" == "1" ]] && { echo "    ... it1..it${max_iter} with --cov-reference-file the previous iteration (no damping)"; return 0; }
        delta="-"
        [[ -n "${ref}" ]] && delta="$(max_abs_delta "${ref}" "${result}")"
        printf '%s\t%s\t%s\n' "${k}" "${delta}" "$(minima "${dir}/${fit}_${base}_it${k}.log")" >> "${trajectory}"
        if [[ "${delta}" != "-" ]] && awk -v d="${delta}" -v t="${tol}" 'BEGIN { exit !(d < t) }'; then
            echo "converged ${fit} ${base} at it${k}"
            return 0
        fi
        ref="${result}"
    done
    echo "not converged ${fit} ${base} after it${max_iter}"
}

recheck_toy() {
    local ens="$1" seed="$2"
    local fit="${ENSEMBLE_FIT[${ens}]}" dir="${study_output}/${ENSEMBLE_FIT[${ens}]}"
    local name="${fit}_${ens}_iterfrozen_s${seed}"
    [[ -f "${dir}/${name}.converged" ]] || return 0
    [[ -s "${dir}/${fit}_${ens}_recheck_s${seed}_global_fit.txt" ]] && return 0
    toy_fit "${ens}" "${seed}" "${ens}_recheck_s${seed}" "${dir}/${name}_reference.txt" \
        || { echo "FAILED ${fit} ${ens}_recheck_s${seed}" >&2; return 1; }
}

run_parallel() {  # run_parallel <function> <"ens seed">...
    local fn="$1"; shift
    local item
    for item in "$@"; do
        if [[ "${dry_run}" == "1" ]]; then
            # shellcheck disable=SC2086
            "${fn}" ${item}
            continue
        fi
        while (($(jobs -rp | wc -l) >= max_jobs)); do wait -n || true; done
        # shellcheck disable=SC2086
        "${fn}" ${item} &
    done
    wait
}

for mode in "${modes[@]}"; do
    case "${mode}" in
        stuck)
            echo "==> stuck toys (${#STUCK[@]}), full search, no damping, at most ${max_iter} refits"
            run_parallel stuck_toy "${STUCK[@]}"
            ;;
        recheck)
            items=()
            for ens in A B10 B11 B12 C00 C12 D; do
                for seed in $(seq "${recheck_seeds%-*}" "${recheck_seeds#*-}"); do items+=("${ens} ${seed}"); done
            done
            echo "==> recheck of converged toys, seeds ${recheck_seeds} of every ensemble (${#items[@]} fits at most)"
            run_parallel recheck_toy "${items[@]}"
            if [[ "${dry_run}" != "1" ]]; then
                echo "    converged toy vs refit with the full search (max |delta| over parameters, chi2 before -> after):"
                for item in "${items[@]}"; do
                    read -r ens seed <<< "${item}"
                    fit="${ENSEMBLE_FIT[${ens}]}"; dir="${study_output}/${fit}"
                    old="${dir}/${fit}_${ens}_iterfrozen_s${seed}_global_fit.txt"; new="${dir}/${fit}_${ens}_recheck_s${seed}_global_fit.txt"
                    [[ -s "${old}" && -s "${new}" ]] || continue
                    printf '    %-4s s%-4s %s  %s -> %s\n' "${ens}" "${seed}" "$(max_abs_delta "${old}" "${new}")" \
                        "$(sed -n 's/.*chi2 = \([^ ]*\).*/\1/p' "${old}")" "$(sed -n 's/.*chi2 = \([^ ]*\).*/\1/p' "${new}")"
                done
            fi
            ;;
        cycle)
            echo "==> cycling toys (${#CYCLE[@]}): ${cycle_iter} more refits with chi2 recorded, then profiles at the two states"
            run_parallel cycle_toy "${CYCLE[@]}"
            if [[ "${dry_run}" != "1" ]]; then
                for item in "${CYCLE[@]}"; do
                    read -r ens seed <<< "${item}"
                    fit="${ENSEMBLE_FIT[${ens}]}"
                    echo "--- ${ens} s${seed}"; cat "${study_output}/${fit}/${fit}_${ens}_cycle_s${seed}_trajectory.txt" 2>/dev/null
                done
            fi
            ;;
        lowpn)
            dir="${output_root}/lowpn_study_fit_results/nuwro/highpn/minerva_k6_uniform_nuisance"
            xml="${script_dir}/../lowpn_study/nuwro/highpn/minerva_k6_uniform_nuisance.xml"
            echo "==> low-p_n nuwro/highpn/minerva_k6_uniform_nuisance: prefix diag_, full search, --seed 1, no damping"
            MAX_ITER="${max_iter}" DRY_RUN="${dry_run}" PROFIT_SEED=1 "${script_dir}/../iterate_frozen.sh" "${dir}" minerva_k6_uniform_nuisance diag_ \
                "${profit_bin}" --xml "${xml}" --tag minerva_k6_uniform_nuisance --nthread 8 --chi2 CNP \
                && echo "converged (${dir}/minerva_k6_uniform_nuisance_diag_trajectory.txt)" \
                || echo "not converged (${dir}/minerva_k6_uniform_nuisance_diag_trajectory.txt)"
            ;;
        *) echo "Unknown mode: ${mode} (stuck, recheck, lowpn)" >&2; exit 2 ;;
    esac
done
