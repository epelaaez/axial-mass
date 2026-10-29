#!/usr/bin/env bash
set -euo pipefail

# Rerun every study that depends on the PROfit input files: the binning study
# (nuwro and asimov fake data, plus nuwro with the CNP chi2), the p_n-range
# (low-p_n) study, the one-dimensional (Q^2-only) study and the injection
# study. Use it after the inputs are regenerated (new ROOT file from
# uboone_ngem/src/save_PROfit_rootfiles_zexp.py) or after a PROfit fix.
#
# PROfit caches the processed inputs as *.bin next to each fit, keyed on a hash
# of the XML that covers the binning, the file/branch names and the systematic
# names, but NOT a systematic's type (spline vs spline_to_covariance), knobvals
# or restrict, and not the input file's contents. A regenerated input file or a
# changed knob definition is therefore NOT picked up by itself: PROfit silently
# reuses the old binaries (seen 2026-10-03, when the binning rerun reproduced the
# old nuisance set and knob grids). This script deletes every cached binary older
# than the newest input file OR the newest study XML before running (--keep-bins
# disables that).
#
# The injection XMLs come from python/notebooks/19_modeling_knob_screen.ipynb
# (knobs.json) and python/scripts/make_injection_study_xmls.py; the binning
# XMLs from python/scripts/make_binning_study_xmls.py, the low-p_n XMLs from
# python/scripts/make_lowpn_study_xmls.py and the one-dimensional XMLs from
# python/scripts/make_oned_study_xmls.py. Regenerate those first when the knob
# list or binnings change.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass}"
log_dir="${output_root}/logs"
dry_run="${DRY_RUN:-0}"

all_studies=(binning lowpn oned injection)
requested_studies=()
keep_bins=0

usage() {
    cat <<'USAGE'
Usage: ./run_studies.sh [options]

Steps, in order:
  1. delete stale PROfit *.bin caches below the study results (older than the inputs or the XMLs)
  2. binning:    xml/binning_study/run_all.sh --family nuwro
                 xml/binning_study/run_all.sh --family asimov
                 xml/binning_study/run_all.sh --family nuwro --chi2 CNP
  3. lowpn:      xml/lowpn_study/run_all.sh --family nuwro
                 xml/lowpn_study/run_all.sh --family asimov
  4. oned:       xml/oned_study/run_all.sh --family nuwro
                 xml/oned_study/run_all.sh --family asimov
  5. injection:  xml/injection_study/run_all.sh

Options:
  --only STUDY       Run only this study (binning, lowpn, oned, injection); repeatable
  --binning-only     Same as --only binning
  --injection-only   Same as --only injection
  --keep-bins        Skip step 1 (reuse every cached binary as is)
  -h, --help         Show this help

Environment (passed through to the run_all.sh scripts):
  JOBS=4 NTHREADS=8 OUTPUT_ROOT PROFIT_BIN DRY_RUN=1 MCMC_ITERATIONS MCMC_BURNIN
Logs go to OUTPUT_ROOT/logs/run_studies_<timestamp>_<step>.log.
USAGE
}

while (($#)); do
    case "$1" in
        --only)
            if (($# < 2)); then
                echo "--only requires a study name (${all_studies[*]})" >&2
                exit 2
            fi
            requested_studies+=("$2")
            shift
            ;;
        --binning-only) requested_studies+=(binning) ;;
        --injection-only) requested_studies+=(injection) ;;
        --keep-bins) keep_bins=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

for requested in "${requested_studies[@]}"; do
    case " ${all_studies[*]} " in
        *" ${requested} "*) ;;
        *) echo "No such study: ${requested} (available: ${all_studies[*]})" >&2; exit 2 ;;
    esac
done
((${#requested_studies[@]})) || requested_studies=("${all_studies[@]}")

selected() {
    # selected <study>: true when the study was requested (or none was, so all run)
    local study
    for study in "${requested_studies[@]}"; do
        [[ "${study}" == "$1" ]] && return 0
    done
    return 1
}

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"

newest_input_mtime() {
    # Newest modification time (epoch seconds) of the given XMLs and of every input file they reference.
    {
        stat -c %Y "$@"
        grep -ho 'filename="[^"]*"' "$@" | cut -d'"' -f2 | sort -u \
            | while IFS= read -r f; do [[ -f "${f}" ]] && stat -c %Y "${f}"; done
    } | sort -n | tail -1
}

delete_stale_bins() {
    # delete_stale_bins <results dir> <xml files...>
    local results="$1"; shift
    [[ -d "${results}" ]] || return 0
    local mtime
    mtime="$(newest_input_mtime "$@")"
    [[ -n "${mtime}" ]] || { echo "  ${results}: no input file found, keeping the caches"; return 0; }
    local stale
    mapfile -t stale < <(find "${results}" -name '*.bin' ! -newermt "@${mtime}")
    echo "  ${results}: ${#stale[@]} cached binaries older than the inputs/XMLs ($(date -d "@${mtime}" '+%F %T'))"
    ((${#stale[@]})) || return 0
    if [[ "${dry_run}" == "1" ]]; then
        printf '    would delete %s\n' "${stale[@]:0:3}"
        ((${#stale[@]} > 3)) && echo "    ... and $((${#stale[@]} - 3)) more"
    else
        rm -f "${stale[@]}"
    fi
}

step() {
    # step <name> <command...>: run and tee to its own log
    local name="$1"; shift
    local log="${log_dir}/run_studies_${timestamp}_${name}.log"
    echo "==> [$(date '+%F %T')] ${name}: $*  (log: ${log})"
    "$@" 2>&1 | tee "${log}"
}

if [[ "${keep_bins}" != "1" ]]; then
    echo "==> [$(date '+%F %T')] stale PROfit caches"
    for study in "${requested_studies[@]}"; do
        delete_stale_bins "${output_root}/${study}_study_fit_results" "${script_dir}/${study}_study"/*/*/*.xml
    done
fi

if selected binning; then
    step binning_nuwro "${script_dir}/binning_study/run_all.sh" --family nuwro
    step binning_asimov "${script_dir}/binning_study/run_all.sh" --family asimov
    step binning_nuwro_cnp "${script_dir}/binning_study/run_all.sh" --family nuwro --chi2 CNP
fi
if selected lowpn; then
    step lowpn_nuwro "${script_dir}/lowpn_study/run_all.sh" --family nuwro
    step lowpn_asimov "${script_dir}/lowpn_study/run_all.sh" --family asimov
fi
if selected oned; then
    step oned_nuwro "${script_dir}/oned_study/run_all.sh" --family nuwro
    step oned_asimov "${script_dir}/oned_study/run_all.sh" --family asimov
fi
if selected injection; then
    step injection "${script_dir}/injection_study/run_all.sh"
fi
echo "==> [$(date '+%F %T')] done"
