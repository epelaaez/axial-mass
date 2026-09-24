#!/usr/bin/env bash
set -euo pipefail

# Run the finer-binning study: every XML below xml/binning_study/<family>/<variant>/.
# The XMLs are generated from xml/<family> by python/scripts/make_binning_study_xmls.py;
# regenerate them there rather than editing them here.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass}"
study_output="${output_root}/binning_study_fit_results"
profit_bin="${PROFIT_BIN:-/nevis/riverside/share/epelaez/PROfit/build/bin/PROfit}"
nthreads="${NTHREADS:-8}"
stages="${STAGES:-plot profile}"
dry_run="${DRY_RUN:-0}"
plot_with_splines="${PLOT_WITH_SPLINES:-1}"
plot_with_covar="${PLOT_WITH_COVAR:-0}"
fit_name="${FIT:-}"
variant_name="${VARIANTS:-}"
family_name="${FAMILIES:-}"
chi2="${CHI2:-}"
mcmc_iterations="${MCMC_ITERATIONS:-}"
mcmc_burnin="${MCMC_BURNIN:-}"

usage() {
    cat <<'USAGE'
Usage: ./run_all.sh [options]

Run every XML below xml/binning_study/<family>/<variant>/ (families: nuwro,
asimov; variants: nominal, fine_q2, fine_pn, fine_both, ...). Results are
written to OUTPUT_ROOT/binning_study_fit_results/<family>/<variant>/<fit>/.

Options:
  --family NAME                Run only this XML family (repeatable)
  --variant NAME               Run only this binning variant (repeatable)
  --fit NAME                   Run only this fit name (repeatable)
  --chi2 NAME                  PROfit chi2 statistic (neyman, pearson, CNP,
                               poisson). Anything but the default neyman is
                               passed as --chi2 and the results go to the
                               family directory <family>_<chi2>, so they never
                               overwrite the default-statistic results.
  -h, --help                   Show this help

Environment:
  FAMILIES="nuwro asimov"       Same as repeated --family
  VARIANTS="fine_q2 fine_both"  Same as repeated --variant
  FIT="minerva_k6 lqcd_k6"      Same as repeated --fit
  CHI2=CNP                      Same as --chi2
  OUTPUT_ROOT, PROFIT_BIN, NTHREADS, STAGES, DRY_RUN, PLOT_WITH_SPLINES,
  MCMC_ITERATIONS, MCMC_BURNIN behave as in xml/run_all.sh.
  PLOT_WITH_COVAR defaults to 0 here (covariance PDFs are slow at many bins);
  set PLOT_WITH_COVAR=1 to write them.
USAGE
}

requested_fits=()
if [[ -n "${fit_name}" ]]; then
    read -r -a fit_list <<< "${fit_name}"
    requested_fits+=("${fit_list[@]}")
fi
requested_variants=()
if [[ -n "${variant_name}" ]]; then
    read -r -a variant_list <<< "${variant_name}"
    requested_variants+=("${variant_list[@]}")
fi
requested_families=()
if [[ -n "${family_name}" ]]; then
    read -r -a family_list <<< "${family_name}"
    requested_families+=("${family_list[@]}")
fi

while (($#)); do
    case "$1" in
        --family)
            if (($# < 2)); then
                echo "--family requires a family name" >&2
                exit 2
            fi
            requested_families+=("$2")
            shift
            ;;
        --chi2)
            if (($# < 2)); then
                echo "--chi2 requires a statistic name" >&2
                exit 2
            fi
            chi2="$2"
            shift
            ;;
        --variant)
            if (($# < 2)); then
                echo "--variant requires a variant name" >&2
                exit 2
            fi
            requested_variants+=("$2")
            shift
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
for index in "${!requested_variants[@]}"; do
    requested_variants[index]="${requested_variants[index]%/}"
    requested_variants[index]="${requested_variants[index]##*/}"
done

# Every subdirectory holding variant subdirectories with XMLs is a family.
families=()
shopt -s nullglob
for family_dir in "${script_dir}"/*/; do
    family_dir="${family_dir%/}"
    family_xmls=("${family_dir}"/*/*.xml)
    ((${#family_xmls[@]})) || continue
    families+=("$(basename -- "${family_dir}")")
done
shopt -u nullglob

if ((${#families[@]} == 0)); then
    echo "No family/variant directories with XML files below ${script_dir}." >&2
    echo "Generate them with python/scripts/make_binning_study_xmls.py." >&2
    exit 1
fi

if ((${#requested_families[@]})); then
    selected_families=()
    for requested in "${requested_families[@]}"; do
        requested="${requested%/}"
        requested="${requested##*/}"
        matched=0
        for family in "${families[@]}"; do
            if [[ "${family}" == "${requested}" ]]; then
                selected_families+=("${family}")
                matched=1
            fi
        done
        if [[ "${matched}" == "0" ]]; then
            echo "No such XML family: ${requested} (available: ${families[*]})" >&2
            exit 2
        fi
    done
    families=("${selected_families[@]}")
fi

# Variants: the union over the selected families, in name order; --variant filters it.
declare -A variant_seen=()
variants=()
for family in "${families[@]}"; do
    shopt -s nullglob
    for variant_dir in "${script_dir}/${family}"/*/; do
        variant_dir="${variant_dir%/}"
        variant_xmls=("${variant_dir}"/*.xml)
        ((${#variant_xmls[@]})) || continue
        variant="$(basename -- "${variant_dir}")"
        if [[ -z "${variant_seen[${variant}]:-}" ]]; then
            variant_seen[${variant}]=1
            variants+=("${variant}")
        fi
    done
    shopt -u nullglob
done

if ((${#requested_variants[@]})); then
    selected_variants=()
    for requested in "${requested_variants[@]}"; do
        matched=0
        for variant in "${variants[@]}"; do
            if [[ "${variant}" == "${requested}" ]]; then
                selected_variants+=("${variant}")
                matched=1
            fi
        done
        if [[ "${matched}" == "0" ]]; then
            echo "No such binning variant: ${requested} (available: ${variants[*]})" >&2
            exit 2
        fi
    done
    variants=("${selected_variants[@]}")
fi

# The chi2 statistic: neyman is PROfit's default and keeps the plain family directory.
chi2_args=()
chi2_suffix=""
if [[ -n "${chi2}" && "${chi2,,}" != "neyman" ]]; then
    chi2_args=(--chi2 "${chi2}")
    chi2_suffix="_${chi2,,}"
fi

if [[ ! -x "${profit_bin}" ]]; then
    echo "PROfit executable is not available: ${profit_bin}" >&2
    exit 1
fi

xml_files=()
for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
        shopt -s nullglob
        variant_xmls=("${script_dir}/${family}/${variant}"/*.xml)
        shopt -u nullglob
        xml_files+=("${variant_xmls[@]}")
    done
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
            echo "No enabled family/variant contains: ${requested_fit}.xml" >&2
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

mkdir -p "${study_output}"

for family in "${families[@]}"; do
  family_output="${study_output}/${family}${chi2_suffix}"
  echo "==> ${family}${chi2_suffix}"
  for variant in "${variants[@]}"; do
    variant_dir="${script_dir}/${family}/${variant}"
    [[ -d "${variant_dir}" ]] || continue
    variant_output="${family_output}/${variant}"
    mkdir -p "${variant_output}"

    shopt -s nullglob
    variant_xmls=("${variant_dir}"/*.xml)
    shopt -u nullglob

    echo "  -> ${variant}"
    for xml in "${variant_xmls[@]}"; do
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
        fit_output="${variant_output}/${fit}"
        mkdir -p "${fit_output}"

        echo "    ${fit}"
        for stage in "${stage_list[@]}"; do
            echo "        ${stage}"
            global_stage_args=("${chi2_args[@]}")
            subcommand_args=()
            if [[ "${stage}" == "plot" && "${plot_with_splines}" == "1" ]]; then
                subcommand_args+=(--with-splines)
            fi
            # Off by default here: the covariance PDFs dominate the plot stage at
            # a few hundred bins and are not needed to compare fits. PLOT_WITH_COVAR=1
            # restores them (needed by 12_spline_factorization_validation.ipynb).
            if [[ "${stage}" == "plot" && "${plot_with_covar}" == "1" ]]; then
                subcommand_args+=(--with-covar)
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
done
