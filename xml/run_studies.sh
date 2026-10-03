#!/usr/bin/env bash
set -euo pipefail

# Rerun every study that depends on the PROfit input files: the binning study
# (nuwro and asimov fake data, plus nuwro with the CNP chi2) and the injection
# study. Use it after the inputs are regenerated (new ROOT file from
# uboone_ngem/src/save_PROfit_rootfiles_zexp.py) or after a PROfit fix.
#
# PROfit caches the processed inputs as *.bin next to each fit, keyed on the XML
# only, so a regenerated input file is NOT picked up by itself. This script
# deletes every cached binary older than the newest input the study XMLs point
# at before running (--keep-bins disables that).
#
# The injection XMLs come from python/notebooks/19_modeling_knob_screen.ipynb
# (knobs.json) and python/scripts/make_injection_study_xmls.py; the binning
# XMLs from python/scripts/make_binning_study_xmls.py. Regenerate those first
# when the knob list or binnings change.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass}"
log_dir="${output_root}/logs"
dry_run="${DRY_RUN:-0}"

run_binning=1
run_injection=1
keep_bins=0

usage() {
    cat <<'USAGE'
Usage: ./run_studies.sh [options]

Steps, in order:
  1. delete stale PROfit *.bin caches below the study results (older than the inputs)
  2. xml/binning_study/run_all.sh --family nuwro
  3. xml/binning_study/run_all.sh --family asimov
  4. xml/binning_study/run_all.sh --family nuwro --chi2 CNP
  5. xml/injection_study/run_all.sh

Options:
  --binning-only     Steps 1-4
  --injection-only   Steps 1 and 5
  --keep-bins        Skip step 1 (reuse every cached binary as is)
  -h, --help         Show this help

Environment (passed through to the run_all.sh scripts):
  JOBS=4 NTHREADS=8 OUTPUT_ROOT PROFIT_BIN DRY_RUN=1
Logs go to OUTPUT_ROOT/logs/run_studies_<timestamp>_<step>.log.
USAGE
}

while (($#)); do
    case "$1" in
        --binning-only) run_injection=0 ;;
        --injection-only) run_binning=0 ;;
        --keep-bins) keep_bins=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"

newest_input_mtime() {
    # Newest modification time (epoch seconds) of every input file the given XMLs reference.
    grep -ho 'filename="[^"]*"' "$@" | cut -d'"' -f2 | sort -u \
        | while IFS= read -r f; do [[ -f "${f}" ]] && stat -c %Y "${f}"; done | sort -n | tail -1
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
    echo "  ${results}: ${#stale[@]} cached binaries older than the inputs ($(date -d "@${mtime}" '+%F %T'))"
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
    if [[ "${run_binning}" == "1" ]]; then
        delete_stale_bins "${output_root}/binning_study_fit_results" "${script_dir}"/binning_study/*/*/*.xml
    fi
    if [[ "${run_injection}" == "1" ]]; then
        delete_stale_bins "${output_root}/injection_study_fit_results" "${script_dir}"/injection_study/*/*/*.xml
    fi
fi

if [[ "${run_binning}" == "1" ]]; then
    step binning_nuwro "${script_dir}/binning_study/run_all.sh" --family nuwro
    step binning_asimov "${script_dir}/binning_study/run_all.sh" --family asimov
    step binning_nuwro_cnp "${script_dir}/binning_study/run_all.sh" --family nuwro --chi2 CNP
fi
if [[ "${run_injection}" == "1" ]]; then
    step injection "${script_dir}/injection_study/run_all.sh"
fi
echo "==> [$(date '+%F %T')] done"
