#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass}"
profit_bin="${PROFIT_BIN:-/nevis/riverside/share/epelaez/PROfit/build/bin/PROfit}"
nthreads="${NTHREADS:-8}"
stages="${STAGES:-plot profile}"
dry_run="${DRY_RUN:-0}"
plot_with_splines="${PLOT_WITH_SPLINES:-1}"
fit_name="${FIT:-}"
mcmc_iterations="${MCMC_ITERATIONS:-}"
mcmc_burnin="${MCMC_BURNIN:-}"

run_nuwro="${RUN_NUWRO:-1}"
run_asimov="${RUN_ASIMOV:-1}"
run_opendata="${RUN_OPENDATA:-1}"

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [options]

Run every XML in the nuwro, asimov, and opendata subdirectories.

Options:
  --no-nuwro                   Disable nuwro
  --no-asimov                  Disable asimov
  --no-opendata                Disable opendata
  --fit NAME                   Run only this fit name (repeatable)
  -h, --help                   Show this help

The same groups can be disabled with RUN_NUWRO=0, RUN_ASIMOV=0, or
RUN_OPENDATA=0. OUTPUT_ROOT is the common target root; each XML family and fit
is written to OUTPUT_ROOT/<family>_fit_results/<fit>/. FIT accepts one or more
space-separated XML filenames or stems. This is equivalent to repeated --fit
options.

MCMC_ITERATIONS and MCMC_BURNIN are passed to PROfit's profile stage as
MCMC-Iterations and MCMC-Burnin fit options. If unset, PROfit uses its defaults.
EOF
}

requested_fits=()
if [[ -n "${fit_name}" ]]; then
    read -r -a fit_list <<< "${fit_name}"
    requested_fits+=("${fit_list[@]}")
fi

while (($#)); do
    case "$1" in
        --no-nuwro)
            run_nuwro=0
            ;;
        --no-asimov)
            run_asimov=0
            ;;
        --no-opendata)
            run_opendata=0
            ;;
        --fit)
            if (($# < 2)); then
                echo "--fit requires a fit name" >&2
                exit 2
            fi
            requested_fits+=("$2")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

for index in "${!requested_fits[@]}"; do
    requested_fits[index]="${requested_fits[index]##*/}"
    requested_fits[index]="${requested_fits[index]%.xml}"
done

families=()
[[ "${run_nuwro}" == "1" ]] && families+=(nuwro)
[[ "${run_asimov}" == "1" ]] && families+=(asimov)
[[ "${run_opendata}" == "1" ]] && families+=(opendata)

if ((${#families[@]} == 0)); then
    echo "All XML families are disabled; nothing to run."
    exit 0
fi

if [[ ! -x "${profit_bin}" ]]; then
    echo "PROfit executable is not available: ${profit_bin}" >&2
    exit 1
fi

xml_files=()
for family in "${families[@]}"; do
    family_dir="${script_dir}/${family}"
    if [[ ! -d "${family_dir}" ]]; then
        echo "XML family directory does not exist: ${family_dir}" >&2
        exit 1
    fi

    shopt -s nullglob
    family_xmls=("${family_dir}"/*.xml)
    shopt -u nullglob
    if ((${#family_xmls[@]} == 0)); then
        echo "No XML files found in: ${family_dir}" >&2
        exit 1
    fi
    xml_files+=("${family_xmls[@]}")
done

if ((${#requested_fits[@]})); then
    selected_xmls=()
    for requested_fit in "${requested_fits[@]}"; do
        matched=0
        for xml in "${xml_files[@]}"; do
            if [[ "$(basename -- "${xml}" .xml)" == "${requested_fit}" ]]; then
                selected_xmls+=("${xml}")
                matched=1
            fi
        done
        if [[ "${matched}" == "0" ]]; then
            echo "No enabled XML family contains: ${requested_fit}.xml" >&2
            exit 2
        fi
    done
    xml_files=("${selected_xmls[@]}")
fi

missing_inputs=()
while IFS= read -r input_file; do
    [[ -f "${input_file}" ]] || missing_inputs+=("${input_file}")
done < <(
    grep -ho 'filename="[^"]*"' "${xml_files[@]}" \
        | cut -d'"' -f2 \
        | sort -u
)

if ((${#missing_inputs[@]})); then
    echo "The following configured inputs do not exist yet:" >&2
    printf '  %s\n' "${missing_inputs[@]}" >&2
    if [[ "${dry_run}" != "1" ]]; then
        exit 1
    fi
fi

read -r -a stage_list <<< "${stages}"
if ((${#stage_list[@]} == 0)); then
    echo "STAGES does not contain any stages to run." >&2
    exit 1
fi

mkdir -p "${output_root}"

for family in "${families[@]}"; do
    family_dir="${script_dir}/${family}"
    family_output="${output_root}/${family}_fit_results"
    mkdir -p "${family_output}"

    shopt -s nullglob
    family_xmls=("${family_dir}"/*.xml)
    shopt -u nullglob

    echo "==> ${family}"
    for xml in "${family_xmls[@]}"; do
        fit="$(basename -- "${xml}" .xml)"
        if ((${#requested_fits[@]})); then
            selected=0
            for requested_fit in "${requested_fits[@]}"; do
                if [[ "${fit}" == "${requested_fit}" ]]; then
                    selected=1
                    break
                fi
            done
            [[ "${selected}" == "1" ]] || continue
        fi
        fit_output="${family_output}/${fit}"
        mkdir -p "${fit_output}"

        echo "    ${fit}"
        for stage in "${stage_list[@]}"; do
            echo "        ${stage}"
            global_stage_args=()
            subcommand_args=()
            if [[ "${stage}" == "plot" && "${plot_with_splines}" == "1" ]]; then
                subcommand_args+=(--with-splines)
            fi
            if [[ "${stage}" == "profile" ]]; then
                if [[ -n "${mcmc_iterations}" ]]; then
                    global_stage_args+=(--fit-options MCMC-Iterations "${mcmc_iterations}")
                fi
                if [[ -n "${mcmc_burnin}" ]]; then
                    global_stage_args+=(--fit-options MCMC-Burnin "${mcmc_burnin}")
                fi
            fi

            if [[ "${dry_run}" == "1" ]]; then
                printf '        (cd %q && %q --xml %q --tag %q --output v1 --nthread %q --log %q --progress' \
                    "${fit_output}" "${profit_bin}" "${xml}" "${fit}" "${nthreads}" \
                    "${stage}.log"
                if ((${#global_stage_args[@]})); then
                    printf ' %q' "${global_stage_args[@]}"
                fi
                printf ' %q' "${stage}"
                if ((${#subcommand_args[@]})); then
                    printf ' %q' "${subcommand_args[@]}"
                fi
                printf ')\n'
                continue
            fi

            (
                cd "${fit_output}"
                "${profit_bin}" \
                    --xml "${xml}" \
                    --tag "${fit}" \
                    --output v1 \
                    --nthread "${nthreads}" \
                    --log "${stage}.log" \
                    --progress \
                    "${global_stage_args[@]}" \
                    "${stage}" \
                    "${subcommand_args[@]}"
            )
        done
    done
done
