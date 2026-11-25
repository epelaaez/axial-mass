#!/usr/bin/env bash
set -euo pipefail

# Iterated frozen-covariance reference for one fit; called by the runners (COV=iterfrozen) before
# their profile stage. Needs PROfit develop >= 9e9d61c
# (--frozen-cov is an alias of --fixed-cov since 88c996a).
#
#   iterate_frozen.sh <dir> <tag> <prefix> <PROfit command with its global arguments...>
#
# runs, in <dir>,
#   <command> --frozen-cov --output <prefix>it0 global                                (M frozen at the CV)
#   <command> --frozen-cov --output <prefix>it<k> --cov-reference-file <it k-1> global
# until no parameter moves by more than TOL (units of global_fit.txt: prior sigma for splines)
# between two consecutive iterations, and copies the last best fit to
#   <dir>/<tag>_<prefix>cov_reference.txt
# which the caller passes as --frozen-cov --cov-reference-file to its profile stage.
# The command must carry --xml, --tag <tag>, --nthread and whatever defines the fit (--chi2,
# --inject-systs, --fix, ...), but not --output, --log or a subcommand.
# <dir>/<tag>_<prefix>trajectory.txt lists iteration, chi2 and max |delta|. Nothing is rerun when the
# reference file is already there; unfinished iterations are resumed.
#
#   iterate_frozen.sh --check <dir> <tag> <prefix> <output>
#
# compares <tag>_<output>_global_fit.txt (the profile stage's own best fit) with the reference and
# warns when they differ by more than TOL: the profile scan then found a lower minimum than the
# iterations did, and the fit should be iterated again from there.
#
# Search (2026-11-24): every fit passes --seed PROFIT_SEED (default 1) and --fit-options
# latin_phys_only 0. PROfit's grad-* presets otherwise put every Latin-hypercube start at the origin in
# an all-spline model (nparams = 0), so the global fit tries two starting points only and can report
# either of two near-degenerate minima depending on the reference; without --seed the swarm step also
# changes from run to run.
# The command must not carry --seed or --fit-options itself.
# A fit that still does not converge within MAX_ITER (in the 2026-11-24 diagnostics: a genuine two-point
# cycle, frozen at either point the best fit is the other) is reported as not converged.
#
# Environment: TOL=0.01, MAX_ITER=30, PROFIT_SEED=1, DRY_RUN=1.

tol="${TOL:-0.01}"
max_iter="${MAX_ITER:-30}"
dry_run="${DRY_RUN:-0}"
frozen_args=(--seed "${PROFIT_SEED:-1}" --fit-options latin_phys_only 0 --frozen-cov)

# max_abs_delta <previous global_fit.txt> <current global_fit.txt>: largest |change| over the
# parameters, matched by position (plotnames need not be unique). Fails if the lists differ.
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

if [[ "${1:-}" == "--check" ]]; then
    (($# == 5)) || { echo "usage: iterate_frozen.sh --check <dir> <tag> <prefix> <output>" >&2; exit 2; }
    dir="$2" tag="$3" prefix="$4" output="$5"
    reference="${dir}/${tag}_${prefix}cov_reference.txt"
    result="${dir}/${tag}_${output}_global_fit.txt"
    [[ -s "${reference}" && -s "${result}" ]] || { echo "WARNING ${dir}: cannot check ${output} against the covariance reference (file missing)" >&2; exit 0; }
    delta="$(max_abs_delta "${reference}" "${result}")" || { echo "WARNING ${dir}: ${output} and the covariance reference list different parameters" >&2; exit 0; }
    if ! awk -v d="${delta}" -v t="${tol}" 'BEGIN { exit !(d < t) }'; then
        echo "WARNING ${dir}: best fit of ${output} is ${delta} sigma from its covariance reference (TOL ${tol}); iterate again from it" >&2
        echo "${delta}" > "${dir}/${tag}_${prefix}reference_mismatch.txt"
    else
        rm -f "${dir}/${tag}_${prefix}reference_mismatch.txt"
    fi
    exit 0
fi

(($# >= 4)) || { echo "usage: iterate_frozen.sh <dir> <tag> <prefix> <PROfit command...>" >&2; exit 2; }
dir="$1" tag="$2" prefix="$3"
shift 3
command=("$@")
final="${dir}/${tag}_${prefix}cov_reference.txt"
trajectory="${dir}/${tag}_${prefix}trajectory.txt"

if [[ "${dry_run}" == "1" ]]; then
    printf '(cd %q &&' "${dir}"; printf ' %q' "${command[@]}" "${frozen_args[@]}" --output "${prefix}it0" --log "${prefix}it0.log" global; printf ')\n'
    echo "    ... then with --output ${prefix}it<k> --cov-reference-file ${tag}_${prefix}it<k-1>_global_fit.txt until max |delta| < ${tol} (at most it${max_iter}) -> $(basename -- "${final}")"
    exit 0
fi
[[ -s "${final}" ]] && exit 0

printf 'iteration\tchi2\tmax_abs_delta\n' > "${trajectory}"
reference=""
for ((k = 0; k <= max_iter; k++)); do
    out="${prefix}it${k}"
    result="${dir}/${tag}_${out}_global_fit.txt"
    if [[ ! -s "${result}" ]]; then
        reference_args=()
        [[ -n "${reference}" ]] && reference_args=(--cov-reference-file "${reference}")
        (cd "${dir}" && "${command[@]}" "${frozen_args[@]}" --output "${out}" --log "${out}.log" "${reference_args[@]}" global > "${out}.out" 2>&1) \
            || { echo "FAILED ${dir} ${out} (see ${out}.out)" >&2; exit 1; }
        [[ -s "${result}" ]] || { echo "FAILED ${dir} ${out}: no $(basename -- "${result}")" >&2; exit 1; }
        # The iterations are only needed for their best fit: drop the PROglobal plots and chains.
        rm -f "${dir}/${tag}_${out}_PROglobal"*
    fi
    chi2="$(grep -a 'Global Best Fit chi.2:' "${dir}/${out}.log" | tail -n 1 | sed 's/.*chi^2: *//')"
    delta="-"
    if [[ -n "${reference}" ]]; then
        delta="$(max_abs_delta "${reference}" "${result}")" || { echo "FAILED ${dir} ${out}: parameter lists differ" >&2; exit 1; }
    fi
    printf '%s\t%s\t%s\n' "${k}" "${chi2:-nan}" "${delta}" >> "${trajectory}"
    if [[ "${delta}" != "-" ]] && awk -v d="${delta}" -v t="${tol}" 'BEGIN { exit !(d < t) }'; then
        cp "${result}" "${final}"
        exit 0
    fi
    reference="${result}"
done
echo "NOT CONVERGED ${dir} ${tag} ${prefix}: max |delta| ${delta} after it${max_iter} (see $(basename -- "${trajectory}"))" >&2
exit 1
