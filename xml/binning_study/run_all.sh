#!/usr/bin/env bash
set -euo pipefail

# Run the finer-binning study: every XML below xml/binning_study/<family>/<variant>/.
# The XMLs are generated from xml/<family> by python/scripts/make_binning_study_xmls.py;
# regenerate them there rather than editing them here.
#
# Other studies with the same <family>/<variant>/<fit>.xml layout reuse this runner by
# setting STUDY_DIR (their XML tree) and, optionally, STUDY_NAME (default: the tree's
# basename): results go to OUTPUT_ROOT/<STUDY_NAME>_fit_results/. See xml/lowpn_study/run_all.sh.

script_dir="${STUDY_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
study_name="${STUDY_NAME:-$(basename -- "${script_dir}")}"
generator="${STUDY_GENERATOR:-python/scripts/make_${study_name}_xmls.py}"
output_root="${OUTPUT_ROOT:-/nevis/hopper/data/epelaez/axial_mass_cnpmu_iterfrozen}"
study_output="${output_root}/${study_name}_fit_results"
profit_bin="${PROFIT_BIN:-/nevis/riverside/share/epelaez/PROfit/build/bin/PROfit}"
nthreads="${NTHREADS:-8}"
stages="${STAGES:-plot profile}"
dry_run="${DRY_RUN:-0}"
plot_with_splines="${PLOT_WITH_SPLINES:-1}"
plot_with_covar="${PLOT_WITH_COVAR:-0}"
fit_name="${FIT:-}"
variant_name="${VARIANTS:-}"
family_name="${FAMILIES:-}"
chi2="${CHI2:-CNP}"
mcmc_iterations="${MCMC_ITERATIONS:-}"
mcmc_burnin="${MCMC_BURNIN:-}"
max_jobs="${JOBS:-4}"

usage() {
    cat <<'USAGE'
Usage: ./run_all.sh [options]

Run every XML below xml/binning_study/<family>/<variant>/ (families: nuwro,
asimov; variants: nominal, count100, count50, ...). Results are
written to OUTPUT_ROOT/binning_study_fit_results/<family>/<variant>/<fit>/.
Different XMLs run in parallel, JOBS at a time; each XML runs its stages
in sequence so the binaries processed by the first stage are reused.

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
  JOBS=4                        XMLs fitted at the same time (each runs its stages in sequence)
  FAMILIES="nuwro asimov"       Same as repeated --family
  VARIANTS="count50 count10"    Same as repeated --variant
  FIT="minerva_k6 lqcd_k6"      Same as repeated --fit
  CHI2=CNP                      Same as --chi2 (CNP is the default; another statistic adds
                                _<chi2> to the family directory)
  FROZEN_COV=1                  Pass --frozen-cov (M frozen at the CV); results go to
                                <family>_frozencov
  COV=iterfrozen                Default: profile stage with the covariance frozen at the
                                iterated best fit (xml/iterate_frozen.sh; TOL, MAX_ITER).
                                COV=dynamic rebuilds it at every point: <family>_dynamic
  PLAIN_DIRS=1                  No suffix on the family directory whatever the statistic and
                                covariance treatment: use it with a separate OUTPUT_ROOT (e.g. the
                                Neyman / dynamic tree /nevis/hopper/data/epelaez/axial_mass)
  OUTPUT_ROOT, PROFIT_BIN, NTHREADS, STAGES, DRY_RUN, PLOT_WITH_SPLINES,
  MCMC_ITERATIONS, MCMC_BURNIN behave as in xml/run_all.sh.
  STUDY_DIR, STUDY_NAME               run another study tree with this runner (see the header)
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
    echo "Generate them with ${generator}." >&2
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

# The analysis default (since 2026-11-24) is CNP with the iterated frozen covariance: that
# combination keeps the plain <family> directory, anything else gets a suffix.
chi2_args=()
chi2_suffix=""
[[ "${chi2,,}" != "neyman" ]] && chi2_args=(--chi2 "${chi2}")
[[ "${chi2,,}" != "cnp" ]] && chi2_suffix="_${chi2,,}"
# FROZEN_COV=1: PROfit --frozen-cov with the default reference (covariance held at the CV
# prediction; PROfit develop >= 9e9d61c). Results go to <family>_frozencov. The 2026-11-18
# <family>_fixedcov trees came from the superseded --fixed-cov-cv patch.
if [[ "${FROZEN_COV:-0}" == "1" ]]; then
    chi2_args+=(--frozen-cov)
    chi2_suffix+="_frozencov"
fi

# COV=iterfrozen: the profile stage runs with --frozen-cov at the reference iterated by
# xml/iterate_frozen.sh in the same output directory (default). COV=dynamic: covariance rebuilt
# at every tested point, results go to <family>_dynamic. FROZEN_COV=1 without COV means dynamic
# stages on top of --frozen-cov, as before.
if [[ "${FROZEN_COV:-0}" == "1" ]]; then cov_mode="${COV:-dynamic}"; else cov_mode="${COV:-iterfrozen}"; fi
if [[ "${cov_mode}" != "dynamic" && "${cov_mode}" != "iterfrozen" ]]; then
    echo "Unknown COV=${cov_mode} (dynamic, iterfrozen)" >&2
    exit 2
fi
if [[ "${FROZEN_COV:-0}" == "1" ]]; then
    [[ "${cov_mode}" == "iterfrozen" ]] && chi2_suffix+="_iterfrozen"
elif [[ "${cov_mode}" == "dynamic" ]]; then
    chi2_suffix+="_dynamic"
fi
# PLAIN_DIRS=1: plain <family> directories whatever the statistic and covariance treatment.
[[ "${PLAIN_DIRS:-0}" == "1" ]] && chi2_suffix=""
iterate_script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../iterate_frozen.sh"

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

run_profit_stage() {
    # run_profit_stage <fit_output> <xml> <fit> <stage>
    local fit_output="$1" xml="$2" fit="$3" stage="$4"
    local global_stage_args=("${chi2_args[@]}")
    local subcommand_args=()
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
        # --mcmc-correlations: PROfit develop >= 9e9d61c writes the _mcmc_chain TTree (read by every
        # post-fit notebook) and the post-fit correlation matrix only with this flag.
        subcommand_args+=(--mcmc-correlations)
        # One --fit-options flag: the search fix of xml/iterate_frozen.sh plus the chain settings.
        fit_options=(latin_phys_only 0)
        [[ -n "${mcmc_iterations}" ]] && fit_options+=(MCMC-Iterations "${mcmc_iterations}")
        [[ -n "${mcmc_burnin}" ]] && fit_options+=(MCMC-Burnin "${mcmc_burnin}")
        global_stage_args+=(--seed "${PROFIT_SEED:-1}" --fit-options "${fit_options[@]}")
    fi
    local iterated=0
    if [[ "${stage}" == "profile" && "${cov_mode}" == "iterfrozen" ]]; then
        iterated=1
        "${iterate_script}" "${fit_output}" "${fit}" "" \
            "${profit_bin}" --xml "${xml}" --tag "${fit}" --nthread "${nthreads}" "${chi2_args[@]}" || return 1
        global_stage_args+=(--frozen-cov --cov-reference-file "${fit_output}/${fit}_cov_reference.txt")
    fi
    local cmd=("${profit_bin}" --xml "${xml}" --tag "${fit}" --output v1 --nthread "${nthreads}"
               --log "${stage}.log" "${global_stage_args[@]}" "${stage}" "${subcommand_args[@]}")
    if [[ "${dry_run}" == "1" ]]; then
        printf '        (cd %q &&' "${fit_output}"
        printf ' %q' "${cmd[@]}"
        printf ')\n'
        return 0
    fi
    # PROfit logs to --log; ROOT chatter and any crash output go to <stage>.out
    (cd "${fit_output}" && "${cmd[@]}" > "${stage}.out" 2>&1) || return 1
    if [[ "${iterated}" == "1" ]]; then
        "${iterate_script}" --check "${fit_output}" "${fit}" "" v1
    fi
}

run_fit() {
    # All stages of one XML, serially, in its own output directory.
    local family="$1" variant="$2" xml="$3"
    local fit stage
    fit="$(basename -- "${xml}" .xml)"
    local fit_output="${study_output}/${family}${chi2_suffix}/${variant}/${fit}"
    [[ "${dry_run}" == "1" ]] || mkdir -p "${fit_output}"
    for stage in "${stage_list[@]}"; do
        run_profit_stage "${fit_output}" "${xml}" "${fit}" "${stage}" || return 1
    done
    [[ "${dry_run}" == "1" ]] || echo "done   ${family}${chi2_suffix}/${variant}/${fit}"
    return 0
}

# One job per XML (all its stages in sequence), JOBS XMLs at a time.
jobs_list=()
for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
        [[ -d "${script_dir}/${family}/${variant}" ]] || continue
        for xml in "${xml_files[@]}"; do
            [[ "${xml}" == "${script_dir}/${family}/${variant}/"* ]] || continue
            jobs_list+=("${family} ${variant} ${xml}")
        done
    done
done

echo "==> ${#jobs_list[@]} XMLs, ${max_jobs} at a time, ${nthreads} threads each, stages: ${stages} -> ${study_output}"
[[ "${dry_run}" == "1" ]] && max_jobs=1

# Each job records its own failure: wait -n reaps jobs, so their statuses cannot be collected later.
failures="$(mktemp)"
trap 'rm -f "${failures}"' EXIT
for job in "${jobs_list[@]}"; do
    read -r family variant xml <<< "${job}"
    while (($(jobs -rp | wc -l) >= max_jobs)); do
        wait -n || true
    done
    [[ "${dry_run}" == "1" ]] || echo "start  ${family}${chi2_suffix}/${variant}/$(basename -- "${xml}" .xml)"
    { run_fit "${family}" "${variant}" "${xml}" || echo "${family}${chi2_suffix}/${variant}/$(basename -- "${xml}" .xml)" >> "${failures}"; } &
done
wait
mapfile -t failed < "${failures}"

if ((${#failed[@]})); then
    echo "Failed (see the logs in their output directories):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
