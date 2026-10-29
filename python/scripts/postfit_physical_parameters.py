"""Translate PROfit pull posteriors into axial-form-factor parameters.

The public entry point is :func:`run_suite`, used by the three compact
notebooks beside the fit XML files.  Transforming the joint chain preserves
correlations and non-Gaussian marginal shapes.

Posterior quantities are read straight from the chains.  The z-expansion XMLs
carry a ``type="spline_cross_quad"`` systematic, so PROfit's own response is
already exact in the cross terms and no post-hoc correction is applied here.
(Before that systematic existed the chains were importance-reweighted instead;
that route survives only in ``python/notebooks/14_cross_branch_validation.ipynb``,
which checks the two agree.)
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import uproot
from IPython.display import display


REPO_ROOT = Path(__file__).resolve().parents[3]
NGEM_SRC = REPO_ROOT / "uboone_ngem" / "src"
if str(NGEM_SRC) not in sys.path:
    sys.path.insert(0, str(NGEM_SRC))

from zexp_reweighting import (  # noqa: E402
    AXIAL_FORM_FACTOR_Q2_ZERO,
    ZEXP_FA_Q2_ZERO,
    ZEXP_T0_GEV2,
    ZEXP_T_CUT_GEV2,
    LQCD_K7_PRIOR,
    LQCD_K6_PRIOR,
    MINERVA_K6_PRIOR,
    MINERVA_K7_PRIOR,
    MINERVA_LEGACY_PRIOR,
    MINERVA_LQCD_K7_PRIOR,
    MINERVA_LQCD_K6_PRIOR,
    complete_zexp_a_values,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _quantile(samples, quantiles):
    """Column-wise quantiles, mid-point (Hazen) convention.

    ``method="hazen"`` places the k-th of n order statistics at (k - 0.5) / n,
    which is the convention every published interval here was produced with;
    numpy's default ``"linear"`` differs by ~1e-3 on these chains.
    """
    return np.quantile(np.asarray(samples, dtype=float), quantiles, axis=0,
                       method="hazen")


DATA_ROOT = Path("/nevis/hopper/data/epelaez/axial_mass")
FIGURE_ROOT = Path(__file__).resolve().parents[2] / "figs"
# Fit outputs live at DATA_ROOT/<data dir>/<fit>/, written by xml/run_all.sh.
# The on-disk directories are the run_all.sh suite names, so SUITE_DATA_DIRS is
# the identity; it stays as a hook for reading a differently-named production
# (it previously redirected to the earlier zexp_prior_fits* outputs).
SUITE_DATA_DIRS = {
    "nuwro_fit_results": "nuwro_fit_results",
    "asimov_fit_results": "asimov_fit_results",
    "opendata_fit_results": "opendata_fit_results",
}

PUBLICATION_RC = {
    "figure.dpi": 140,
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
}
PUBLICATION_FONT_RC = {
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
}

# Keep each source synchronized with the axial-form-factor prior figure in
# 5_publication_plots.ipynb (Okabe-Ito colorblind-safe palette).
FA_SOURCE_COLORS = {
    "deuterium": "#C000C3",
    "deuterium_k6": "#8B008B",
    "minerva_k8": "#E69F00",
    "minerva_k7": "#D55E00",
    "minerva_k6": "#56B4E9",
    "minerva_k6_uniform": "#1607E4",
    "minerva_k6_uniform_nuisance": "#5B4FE8",
    "lqcd_k6": "#009E73",
    "lqcd_k7": "#007A5A",
    "minerva_lqcd_k6": "#CC79A7",
    "minerva_lqcd_k7": "#A85F8C",
    "ma": "#0072B2",
}

# Contour style for plot_distribution_overlay. Each prior gets its own broken
# line so that priors which nearly coincide stay distinguishable where they
# overlap; the posterior is the only solid contour in the figure.
PRIOR_DASH_PATTERNS = (
    (0, (6.5, 3.0)),
    (0, (1.4, 1.9)),
    (0, (6.5, 2.2, 1.4, 2.2)),
    (0, (3.0, 2.0)),
)
CONTOUR_CORE_ALPHA = .16
CONTOUR_LINEWIDTHS = (1.0, 1.8)

REFERENCE_PRIORS = {
    "minerva_k6": MINERVA_K6_PRIOR,
    "lqcd_k6": LQCD_K6_PRIOR,
    "minerva_lqcd_k6": MINERVA_LQCD_K6_PRIOR,
}

# Standalone reference priors that have no fit of their own. The shared keys
# minerva_k6, lqcd_k6, and minerva_lqcd_k6 take their legend text from SPECS.
REFERENCE_SOURCES = {
    "deuterium": ("Deuterium (2016)", 8),
    "deuterium_k6": ("Deuterium (2025)", 6),
}

# Native kmax=8 Deuterium result from Eqs. (31)--(33) of Meyer et al. (2016),
# arXiv:1603.03048.  The kmax=6 constants immediately below are a translation
# of this same result into the common kmax=6 basis; they are not independent
# Deuterium measurements.
DEUTERIUM_K8_FREE = np.array([2.3, -0.6, -3.8, 2.3])
DEUTERIUM_K8_ERRORS = np.sqrt([0.0154, 1.08, 6.54, 7.40])
DEUTERIUM_K8_CORRELATION = np.array([
    [1.000, 0.335, -0.678, 0.611],
    [0.350, 1.000, -0.898, 0.367],
    [-0.678, -0.898, 1.000, -0.685],
    [0.611, 0.367, -0.685, 1.000],
])

# Translation of the native 2016 kmax=8 result above to kmax=6. The
# publication quotes positive F_A; negate every coefficient to match this
# codebase's negative-F_A/GENIE convention.
DEUTERIUM_K6_FREE = -np.array([-2.08493637, 1.89831616])
DEUTERIUM_K6_COVARIANCE = np.array([
    [0.04304942, 0.02482393],
    [0.02482393, 0.13790576],
])
DEUTERIUM_K6_FULL = -np.array([
    0.54264533, -2.08493637, 1.89831616, 2.40319245,
    -5.88979056, 4.14554900, -1.01497601,
])


@dataclass(frozen=True)
class FitSpec:
    key: str
    title: str
    prior: object = None
    profile_labels: tuple = ()
    chain_branches: tuple = ()
    nuisance_labels: tuple = ()
    nuisance_branches: tuple = ()
    uniform_prior: bool = False
    # Legend text: ``source`` names the prior ("MINERvA (2026)", "uniform",
    # "Gaussian $M_A$"); ``variant`` distinguishes fits that share one prior.
    source: str = ""
    variant: str = ""


# Modeling knobs floated as nuisance parameters, as PROfit labels them (plotname) and as
# the chain branches are named. Display names and prior ranges for the corner plots below.
# Ordered by how strongly the data constrain them, which puts the informative
# panels next to the physical parameters in the corner-style figures, with
# the two-point model switches (XSecShape_CCMEC, DecayAngMEC) last.
NUISANCE_LABELS = ("RPA_CCQE", "FrCEx_N", "MFP_N", "XSecShape_CCMEC", "DecayAngMEC")
NUISANCE_BRANCHES = tuple(f"{label}_UBGenie" for label in NUISANCE_LABELS)
NUISANCE_DISPLAY_NAMES = {
    "AxFFCCQEshape": "AxFFCCQEshape pull",
    "NormCCMEC": "NormCCMEC pull",
    "RPA_CCQE": "RPA CCQE pull",
    "XSecShape_CCMEC": "XSecShape CCMEC pull",
    "MFP_N": "MFP N pull",
    "DecayAngMEC": "DecayAngMEC pull",
    "FrCEx_N": "FrCEx N pull",
}
# Ranges the truncated standard-normal prior is drawn over: the two-point model switches
# (AxFFCCQEshape, XSecShape_CCMEC, DecayAngMEC) are restricted to [0, 1] in the XMLs.
NUISANCE_PRIOR_RANGES = {
    "AxFFCCQEshape": (0, 1),
    "NormCCMEC": (-2, 3),
    "RPA_CCQE": (-3, 3),
    "XSecShape_CCMEC": (0, 1),
    "MFP_N": (-3, 3),
    "DecayAngMEC": (0, 1),
    "FrCEx_N": (-3, 3),
}
# Short axis labels for the corner-style figures, keyed by the display names
# above. The long "... pull" names stay the keys of every result array and of
# the notebooks' AXIS_RANGES; the captions map these short labels to the GENIE
# knob names.
NUISANCE_SHORT_LABELS = {
    "AxFFCCQEshape pull": "AxFF shape",
    "NormCCMEC pull": "MEC norm",
    "RPA CCQE pull": "RPA",
    "XSecShape CCMEC pull": "MEC shape",
    "MFP N pull": r"MFP $N$",
    "DecayAngMEC pull": "MEC angle",
    "FrCEx N pull": r"FrCEx $N$",
}
# Hard bounds (the XMLs' restrict=) of the two-point switches, keyed by display
# name. Their densities (posterior and prior alike) are estimated on exactly
# this range with reflection at the edges, and their panels show no padding
# beyond it.
NUISANCE_BOUNDS = {
    "AxFFCCQEshape pull": (0, 1),
    "XSecShape CCMEC pull": (0, 1),
    "DecayAngMEC pull": (0, 1),
}


SPECS = (
    # The M_A fits and the *_nuisance fits float the same five modeling knobs (since 2026-10-03,
    # chosen by the injection study in 19_modeling_knob_screen.ipynb): RPA_CCQE, XSecShape_CCMEC,
    # MFP_N, DecayAngMEC, FrCEx_N. The dipole fit no longer floats AxFFCCQEshape (the former
    # "ma_no_axff" is now "ma"), so the two M_A fits differ only in the M_A prior.
    FitSpec("ma", r"Dipole $M_A$", source=r"Gaussian $M_A$",
            profile_labels=("MACCQE",) + NUISANCE_LABELS,
            chain_branches=("MaCCQE_UBGenie",) + NUISANCE_BRANCHES),
    FitSpec("ma_uniform", r"Dipole $M_A$ without an $M_A$ pull penalty",
            source=r"uniform $M_A$",
            profile_labels=("MACCQE",) + NUISANCE_LABELS,
            chain_branches=("MaCCQE_UBGenie",) + NUISANCE_BRANCHES,
            uniform_prior=True),
    FitSpec("lqcd_k6", r"LQCD (2026), $k_{\max}=6$", LQCD_K6_PRIOR,
            source="LQCD (2026)"),
    FitSpec(
        "lqcd_k6_nuisance",
        r"LQCD (2026), $k_{\max}=6$, fitted nuisances",
        LQCD_K6_PRIOR,
        source="LQCD (2026)", variant="fitted nuisances",
        nuisance_labels=NUISANCE_LABELS,
        nuisance_branches=NUISANCE_BRANCHES,
    ),
    FitSpec("lqcd_k7", r"LQCD (2026), $k_{\max}=7$", LQCD_K7_PRIOR,
            source="LQCD (2026)"),
    FitSpec("minerva_k6", r"MINERvA (2026), $k_{\max}=6$", MINERVA_K6_PRIOR,
            source="MINERvA (2026)"),
    FitSpec(
        "minerva_k6_nuisance",
        r"MINERvA (2026), $k_{\max}=6$, fitted nuisances",
        MINERVA_K6_PRIOR,
        source="MINERvA (2026)", variant="fitted nuisances",
        nuisance_labels=NUISANCE_LABELS,
        nuisance_branches=NUISANCE_BRANCHES,
    ),
    FitSpec("minerva_k6_uniform",
            r"$k_{\max}=6$ uniform prior",
            MINERVA_K6_PRIOR, uniform_prior=True, source="uniform"),
    FitSpec(
        "minerva_k6_uniform_nuisance",
        r"$k_{\max}=6$ uniform prior, fitted nuisances",
        MINERVA_K6_PRIOR, uniform_prior=True,
        source="uniform", variant="fitted nuisances",
        nuisance_labels=NUISANCE_LABELS,
        nuisance_branches=NUISANCE_BRANCHES,
    ),
    FitSpec("minerva_k7", r"MINERvA (2026), $k_{\max}=7$", MINERVA_K7_PRIOR,
            source="MINERvA (2026)"),
    FitSpec("minerva_k8", r"MINERvA (2023), $k_{\max}=8$", MINERVA_LEGACY_PRIOR,
            source="MINERvA (2023)"),
    FitSpec("minerva_lqcd_k6", r"MINERvA + LQCD (2026), $k_{\max}=6$",
            MINERVA_LQCD_K6_PRIOR, source="MINERvA + LQCD (2026)"),
    FitSpec("minerva_lqcd_k7", r"MINERvA + LQCD (2026), $k_{\max}=7$",
            MINERVA_LQCD_K7_PRIOR, source="MINERvA + LQCD (2026)"),
    FitSpec(
        "minerva_lqcd_k6_nuisance",
        r"MINERvA + LQCD (2026), $k_{\max}=6$, fitted nuisances",
        MINERVA_LQCD_K6_PRIOR,
        source="MINERvA + LQCD (2026)", variant="fitted nuisances",
        nuisance_labels=NUISANCE_LABELS,
        nuisance_branches=NUISANCE_BRANCHES,
    ),
)

def format_prior_label(source, kmax=None):
    """Standard legend text for a prior: ``"<Source> prior, $k_max=N$"``.

    The k_max part is omitted when it does not apply (dipole M_A priors).
    """
    label = f"{source[:1].upper()}{source[1:]} prior"
    if kmax is not None:
        label += rf", $k_{{\max}}={kmax}$"
    return label


def format_posterior_label(source=None, kmax=None, variant=""):
    """Standard legend text for a posterior.

    ``"Posterior, $k_max=N$"`` when the figure involves a single prior; pass
    ``source`` to obtain ``"Posterior from <source> prior, $k_max=N$"`` when
    several priors appear in one figure. ``variant`` (for example ``"fitted
    nuisances"``) is appended after a comma.
    """
    label = "Posterior" if source is None else f"Posterior from {source} prior"
    if kmax is not None:
        label += rf", $k_{{\max}}={kmax}$"
    if variant:
        label += f", {variant}"
    return label


def _spec_kmax(spec):
    return None if spec.prior is None else spec.prior.kmax


def prior_label(spec):
    """Legend text for the prior of a fit in ``SPECS``."""
    return format_prior_label(spec.source, _spec_kmax(spec))


def posterior_label(spec, single_prior=True):
    """Legend text for the posterior of a fit in ``SPECS``.

    With ``single_prior=False`` the prior the posterior was obtained from is
    named, which is needed whenever a figure shows more than one prior.
    """
    return format_posterior_label(
        None if single_prior else spec.source, _spec_kmax(spec), spec.variant
    )


REFERENCE_LABELS = {
    **{key: format_prior_label(source, kmax)
       for key, (source, kmax) in REFERENCE_SOURCES.items()},
    **{spec.key: prior_label(spec) for spec in SPECS if spec.key in REFERENCE_PRIORS},
}


def _root_file(suite, key, data_root=None):
    directory = Path(data_root or DATA_ROOT) / SUITE_DATA_DIRS.get(suite, suite) / key
    preferred = directory / f"{key}_v1_PROfile.root"
    if preferred.exists():
        return preferred
    matches = sorted(directory.glob("*_v1_PROfile.root"))
    if len(matches) == 1:
        return matches[0]
    return None


def _fit_coordinates(root_file, profile_labels, chain_branches, burn_in, thin):
    with uproot.open(root_file) as root:
        chains = [k.split(";")[0] for k in root.keys()
                  if k.split(";")[0].endswith("_mcmc_chain")]
        if len(chains) != 1:
            raise RuntimeError(f"Expected one MCMC chain in {root_file}, found {chains}")
        tree = root[chains[0]]
        samples = np.column_stack([tree[b].array(library="np") for b in chain_branches])
        hist = root["global_fit_result"]
        labels, values = list(hist.axis().labels()), hist.values()
        profile = np.asarray([values[labels.index(label)] for label in profile_labels])
    return samples[burn_in::thin], profile


def _zexp_transform(prior):
    """Return central vector and d(a_0..a_kmax)/d(pull)."""
    covariance = np.asarray(prior.covariance)
    if prior.use_pca:
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        order = np.argsort(values)[::-1]
        free_shifts = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
    else:
        free_shifts = np.diag(np.sqrt(np.diag(covariance)))

    free_cv = np.asarray(prior.free_a_values)
    completed_cv = complete_zexp_a_values(
        free_cv, prior.kmax, prior.t0_gev2,
        t_cut_gev2=prior.t_cut_gev2, fa_q2_zero=prior.fa_q2_zero,
    )
    columns = []
    for j in range(len(free_cv)):
        shifted = complete_zexp_a_values(
            free_cv + free_shifts[:, j], prior.kmax, prior.t0_gev2,
            t_cut_gev2=prior.t_cut_gev2, fa_q2_zero=prior.fa_q2_zero,
        )
        columns.append(np.asarray(shifted) - np.asarray(completed_cv))
    matrix = np.column_stack(columns)
    central = np.asarray(prior.full_a_values)
    assert np.allclose(matrix[1:len(free_cv) + 1], free_shifts)
    return central, matrix


def _truncated_standard_normal(rng, size, low, high):
    """Draw standard-normal values within a spline's fitted range."""
    values = rng.normal(size=size)
    outside = (values < low) | (values > high)
    while outside.any():
        values[outside] = rng.normal(size=outside.sum())
        outside = (values < low) | (values > high)
    return values


def _correlation_from_covariance(covariance):
    sigma = np.sqrt(np.diag(covariance))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.outer(sigma, sigma)
    return np.where(np.isfinite(correlation), correlation, 0.0)


def load_fit(spec, suite, burn_in=0, thin=1, n_prior=50_000, seed=2026,
             data_root=None):
    """Load one fit's chain and transform it to physical parameters.

    ``data_root`` overrides :data:`DATA_ROOT` for this call, so a fit tree
    outside the production tree can be read (the variant trees under
    ``/nevis/hopper/data/epelaez/axial_mass_crosstest/<variant>/`` keep the same
    ``<suite>/<fit>/`` layout below their root).

    Returns ``None`` when the fit directory holds no ``*_v1_PROfile.root``.
    """
    root_file = _root_file(suite, spec.key, data_root)
    if root_file is None:
        return None

    is_ma_fit = spec.prior is None
    if is_ma_fit:
        pulls, profile_pull = _fit_coordinates(
            root_file, spec.profile_labels, spec.chain_branches, burn_in, thin
        )
        # MaCCQE's seven knots are 0.8,...,1.4 GeV: pull zero is 1.1 GeV
        # and one pull unit is 0.1 GeV.
        central, sigma = np.array([1.1]), np.array([0.1])
        samples = central + sigma * pulls[:, :1]
        profile = central + sigma * profile_pull[:1]
        rng = np.random.default_rng(seed)
        # Mirrors restrict="-8, 8" on MACCQE in the ma_uniform XMLs, i.e.
        # M_A in [0.3, 1.9] GeV. Keep the two in step or the prior drawn in the
        # corner panels stops matching the prior the chain was sampled under.
        prior_pulls = (rng.uniform(-8.0, 8.0, size=(n_prior, 1))
                       if spec.uniform_prior
                       else rng.normal(size=(n_prior, 1)))
        prior_samples = central + sigma * prior_pulls
        names = ["M_A [GeV]"]
        nuisance_names = NUISANCE_DISPLAY_NAMES
        nuisance_prior_ranges = NUISANCE_PRIOR_RANGES
        nuisance_labels = spec.profile_labels[1:]
        ma_joint_names = ["M_A [GeV]"] + [
            nuisance_names[label] for label in nuisance_labels
        ]
        ma_joint_samples = np.column_stack([samples[:, 0], pulls[:, 1:]])
        ma_joint_prior_samples = np.column_stack(
            [prior_samples[:, 0]] + [
                _truncated_standard_normal(
                    rng, n_prior, *nuisance_prior_ranges[label]
                )
                for label in nuisance_labels
            ]
        )
        ma_joint_profile = np.r_[profile[0], profile_pull[1:]]
    else:
        branches = tuple(spec.prior.variation_branches)
        # XML plot names usually equal the branch suffix.  The two legacy k=8
        # fits add "MinervaK8" to the profile label, however.
        suffixes = tuple(branch.removeprefix("weight_spline_") for branch in branches)
        if spec.key == "minerva_k8":
            labels = tuple(s.replace("FAzexp", "FAzexpMinervaK8") for s in suffixes)
        else:
            labels = suffixes
        all_labels = labels + spec.nuisance_labels
        all_branches = branches + spec.nuisance_branches
        pulls, profile_pull = _fit_coordinates(
            root_file, all_labels, all_branches, burn_in, thin
        )
        central, matrix = _zexp_transform(spec.prior)
        n_axial = matrix.shape[1]
        samples = central + pulls[:, :n_axial] @ matrix.T
        profile = central + matrix @ profile_pull[:n_axial]
        rng = np.random.default_rng(seed)
        prior_pulls = (
            rng.uniform(-5.0, 5.0, size=(n_prior, matrix.shape[1]))
            if spec.uniform_prior
            else rng.normal(size=(n_prior, matrix.shape[1]))
        )
        prior_samples = central + prior_pulls @ matrix.T
        pull_variance = 25.0 / 3.0 if spec.uniform_prior else 1.0
        sigma = np.sqrt(np.diag(pull_variance * matrix @ matrix.T))
        names = [f"a{i}" for i in range(len(central))]

    n_samples = len(samples)
    eta_samples = None if is_ma_fit else np.array(pulls[:, :n_axial], dtype=float)

    posterior_mean = samples.mean(axis=0)
    q16, median, q84 = _quantile(samples, [0.16, 0.50, 0.84])
    summary = pd.DataFrame({
        "prior_central": central, "prior_sigma": sigma,
        "profile_best_fit": profile, "posterior_mean": posterior_mean,
        "posterior_median": median, "q16": q16, "q84": q84,
        "minus_1sigma": median - q16, "plus_1sigma": q84 - median,
    }, index=names)
    summary.index.name = "parameter"
    # a1...a[kmax-4] are the independent physical coefficients.  a0 and the
    # final four coefficients are fixed by F_A(0) and the four sum rules, so
    # omit those derived quantities from joint-parameter diagnostics.
    if is_ma_fit:
        joint_indices = np.array([0])
    else:
        joint_indices = np.arange(1, len(spec.prior.free_a_values) + 1)
    joint_samples = samples[:, joint_indices]
    joint_names = [names[i] for i in joint_indices]
    covariance_matrix = np.atleast_2d(np.cov(joint_samples, rowvar=False))
    covariance = pd.DataFrame(
        covariance_matrix, index=joint_names, columns=joint_names,
    )
    correlation = pd.DataFrame(
        _correlation_from_covariance(covariance_matrix),
        index=joint_names, columns=joint_names,
    )
    # Chain diagnostics on the fit coordinates (every chain branch): the
    # integrated autocorrelation time per coordinate and the effective sample
    # size the density estimates and run_suite's warning go by.
    chain_names = list(spec.chain_branches if is_ma_fit else all_branches)
    autocorrelation_times, effective_samples = _chain_diagnostics(pulls, chain_names)
    result = dict(spec=spec, root_file=root_file, central=central, sigma=sigma,
                  autocorrelation_times=autocorrelation_times,
                  effective_samples=effective_samples,
                  ess_min=min(effective_samples.values()),
                  samples=samples, prior_samples=prior_samples, profile=profile,
                  q16=q16, q84=q84, names=names, summary=summary,
                  covariance=covariance, correlation=correlation,
                  joint_indices=joint_indices, joint_samples=joint_samples,
                  joint_names=joint_names, eta_samples=eta_samples)
    if is_ma_fit:
        result.update(
            ma_joint_samples=ma_joint_samples,
            ma_joint_prior_samples=ma_joint_prior_samples,
            ma_joint_profile=ma_joint_profile,
            ma_joint_names=ma_joint_names,
        )
    elif spec.nuisance_branches:
        nuisance_names = NUISANCE_DISPLAY_NAMES
        nuisance_prior_ranges = NUISANCE_PRIOR_RANGES
        rng = np.random.default_rng(seed)
        result.update(
            zexp_nuisance_samples=np.column_stack(
                [joint_samples, pulls[:, n_axial:]]
            ),
            zexp_nuisance_prior_samples=np.column_stack(
                [prior_samples[:, joint_indices]] + [
                    _truncated_standard_normal(
                        rng, n_prior, *nuisance_prior_ranges[label]
                    )
                    for label in spec.nuisance_labels
                ]
            ),
            zexp_nuisance_profile=np.r_[
                profile[joint_indices], profile_pull[n_axial:]
            ],
            zexp_nuisance_names=joint_names + [
                nuisance_names[label] for label in spec.nuisance_labels
            ],
        )
    return result


def _credible_density_levels(histogram, probabilities=(0.95, 0.68)):
    """Density thresholds enclosing the requested probability masses."""
    density = np.asarray(histogram, dtype=float)
    total = density.sum()
    if total == 0:
        return []
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / total
    thresholds = []
    for probability in probabilities:
        index = min(np.searchsorted(cumulative, probability), len(ordered) - 1)
        thresholds.append(ordered[index])
    return sorted(set(threshold for threshold in thresholds if threshold > 0))


def _reflect_pad(values, radius, axis):
    """Mirror ``values`` by ``radius`` entries at both ends of ``axis``."""
    pad = [(0, 0)] * values.ndim
    pad[axis] = (radius, radius)
    return np.pad(values, pad, mode="symmetric")


def _gaussian_kernel(sigma):
    """Normalised Gaussian kernel of width ``sigma`` bins and its radius."""
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum(), radius


def _smooth_density(samples, bins, bounds=None, smoothing=1.0):
    """Return a lightly smoothed 1D histogram density for line display.

    ``bounds`` marks a hard parameter range (an XML ``restrict``): the
    histogram then covers exactly that range and is mirrored at both ends
    before smoothing, so a posterior piled up against the bound keeps its
    height there instead of being averaged with the empty space beyond it.
    ``smoothing`` scales the kernel width (in bins), see
    :func:`_smoothing_factor`.
    """
    density, edges = np.histogram(samples, bins=max(80, 2 * bins), range=bounds,
                                  density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    kernel, radius = _gaussian_kernel(1.5 * smoothing)
    if bounds is not None:
        smoothed = np.convolve(_reflect_pad(density, radius, 0), kernel, mode="same")
        return centers, smoothed[radius:-radius]
    # Divide by the kernel's overlap with the histogram. Without this the
    # convolution averages in the zeros beyond the first and last bin, which
    # drags the curve towards zero where the samples simply run out.
    coverage = np.convolve(np.ones_like(density), kernel, mode="same")
    return centers, np.convolve(density, kernel, mode="same") / coverage


def _smooth_density_2d(histogram, sigma=1.0, reflect=(False, False)):
    """Apply a small separable Gaussian kernel to a 2D density estimate.

    ``sigma`` is in bins, so a finer binning needs a larger value to hold the
    physical smoothing length fixed. ``reflect`` says per axis (x, then y)
    whether the histogram ends at a hard parameter bound: that axis is then
    mirrored at its ends before smoothing, as in :func:`_smooth_density`.
    """
    kernel, radius = _gaussian_kernel(sigma)
    smoothed = np.asarray(histogram, dtype=float)
    for axis in (0, 1):
        if reflect[axis]:
            padded = _reflect_pad(smoothed, radius, axis)
            smoothed = np.apply_along_axis(
                lambda values: np.convolve(values, kernel, mode="same"), axis, padded
            )
            keep = np.arange(radius, smoothed.shape[axis] - radius)
            smoothed = np.take(smoothed, keep, axis=axis)
        else:
            smoothed = np.apply_along_axis(
                lambda values: np.convolve(values, kernel, mode="same"), axis, smoothed
            )
    return smoothed


def _density_grid_to_edges(xedges, yedges, density):
    """Grid coordinates and density extended from bin centres to bin edges.

    ``contour``/``contourf`` only draw between grid points, so a credible
    region reaching the outermost bin stops half a bin short of the histogram
    boundary. A histogram is piecewise constant across each bin, so repeating
    the boundary bins out to the edges is the faithful reading of it rather
    than an extrapolation, and it is what lets a posterior that piles up
    against a hard parameter bound be drawn flush with that bound.
    """
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    return (
        np.concatenate(([xedges[0]], xcenters, [xedges[-1]])),
        np.concatenate(([yedges[0]], ycenters, [yedges[-1]])),
        np.pad(density, 1, mode="edge"),
    )


def _panel_binning(display_range, values, bins, max_growth=4, bounds=None):
    """Histogram bounds and bin count for one axis of a corner panel.

    The bounds cover the panel, or the contours stop inside it wherever
    ``axis_ranges`` asks for a wider view than the chain reaches, and they
    cover every sample, because discarding samples would renormalize the
    credible levels. The bin count grows with the widened bounds so that the
    bin width stays the one ``bins`` would have given over the samples alone,
    which keeps the contour resolution independent of the requested view.
    A parameter with hard ``bounds`` is histogrammed on exactly those.
    """
    if bounds is not None:
        return tuple(bounds), bins
    low = min(display_range[0], float(values.min()))
    high = max(display_range[1], float(values.max()))
    span = float(values.max() - values.min())
    if span <= 0 or high <= low:
        return (low, high), bins
    widened = int(np.ceil(bins * (high - low) / span))
    return (low, high), int(min(widened, max_growth * bins))


def _bounded_ticks(axis, bounds):
    """Ticks at the ends and the middle of a hard-bounded parameter range."""
    low, high = bounds
    axis.set_major_locator(FixedLocator([low, (low + high) / 2, high]))
    axis.set_minor_locator(AutoMinorLocator(2))


def _style_corner_ticks(ax, x_only=False, x_bounds=None, y_bounds=None):
    """Few, evenly spaced ticks that cannot collide across adjacent panels.

    Three major ticks per axis, pruned at the panel edges; an axis with hard
    ``bounds`` instead shows its two ends and the middle.
    """
    if x_bounds is None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        _bounded_ticks(ax.xaxis, x_bounds)
    if not x_only:
        if y_bounds is None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        else:
            _bounded_ticks(ax.yaxis, y_bounds)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="minor", length=1.8)


def _format_interval(median, lower_error, upper_error):
    """Format a central value and asymmetric errors at useful precision."""
    positive_errors = [error for error in (lower_error, upper_error) if error > 0]
    if not positive_errors:
        return f"{median:.3g}", f"{lower_error:.2g}", f"{upper_error:.2g}"
    exponent = np.floor(np.log10(min(positive_errors)))
    decimals = max(0, int(1 - exponent))
    return tuple(f"{value:.{decimals}f}"
                 for value in (median, lower_error, upper_error))


def _diagnostic_view(result, show_all_coefficients=False):
    """Return the physical parameters selected for joint diagnostics."""
    if show_all_coefficients:
        indices = np.arange(len(result["names"]))
        return result["samples"], result["names"], indices
    return result["joint_samples"], result["joint_names"], result["joint_indices"]


def _parameter_label(name):
    """Return a publication-style parameter label for plot axes and titles.

    Floated knobs get the short labels in :data:`NUISANCE_SHORT_LABELS`; the
    figure captions map those to the GENIE knob names.
    """
    if name.startswith("a") and name[1:].isdigit():
        return rf"$a_{{{name[1:]}}}$"
    if name == "M_A [GeV]":
        return r"$M_{A}\,[\mathrm{GeV}]$"
    if name in NUISANCE_SHORT_LABELS:
        return NUISANCE_SHORT_LABELS[name]
    return name.removesuffix(" pull")


def _autocorrelation_time(chain, window_factor=5.0):
    """Integrated autocorrelation time of one chain coordinate, in steps.

    Sokal's estimate with the automatic window of Goodman and Weare (as in
    emcee): the autocorrelation sum is truncated at the first lag M with
    M >= window_factor * tau(M). Never below 1.
    """
    values = np.asarray(chain, dtype=float)
    values = values - values.mean()
    n = len(values)
    if n < 10 or not np.any(values):
        return 1.0
    spectrum = np.fft.rfft(values, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    inside_window = np.arange(n) < window_factor * taus
    window = int(np.argmin(inside_window)) if np.any(~inside_window) else n - 1
    return float(max(1.0, taus[window]))


def _chain_diagnostics(pulls, names):
    """Autocorrelation time and effective sample size per chain coordinate."""
    n_steps = len(pulls)
    times = {name: _autocorrelation_time(pulls[:, i]) for i, name in enumerate(names)}
    sizes = {name: n_steps / tau for name, tau in times.items()}
    return times, sizes


# Effective sample size at which the density estimates use their nominal
# kernel; shorter chains get a wider one (see _smoothing_factor) and
# run_suite warns below it.
SMOOTHING_REFERENCE_ESS = 2000.0


def _smoothing_factor(result):
    """Kernel-width factor for the density estimates of one fit.

    Grows as (SMOOTHING_REFERENCE_ESS / ESS)^(1/5), the usual bandwidth
    scaling with sample size, from 1 at the reference effective sample size,
    clipped to [1, 2.5]. Fits loaded without chain diagnostics use 1.
    """
    ess = result.get("ess_min")
    if not ess:
        return 1.0
    return float(np.clip((SMOOTHING_REFERENCE_ESS / ess) ** 0.2, 1.0, 2.5))


# All one-column corner PDFs are placed at the same width in the paper. The
# two-parameter corner is the typography reference: every other figure derives
# its font sizes from it through column_font_sizes below.
CORNER_REFERENCE_WIDTH = 4.0


# Font sizes the reference corner draws at CORNER_REFERENCE_WIDTH. Every other
# publication figure is placed at the same printed width, so a wider source
# figure is shrunk more by LaTeX and has to grow these by its own width ratio
# to print at the same size as the corners it sits beside.
COLUMN_REFERENCE_FONT_SIZES = {
    "tick": 7.0,
    "legend": 7.5,
    "title": 9.0,
    "label": 10.0,
}


# The printed widths the paper's two-column REVTeX layout offers: one column,
# which is what a `figure` gets and what the reference corner is placed at, and
# the full text width, which only a `figure*` gets.
COLUMN_WIDTH = 3.4
TEXT_WIDTH = 7.0


def column_font_sizes(figure_width, placed_width=COLUMN_WIDTH):
    """Font sizes that print like the reference corner.

    ``figure_width`` is the source figure's width in inches and
    ``placed_width`` the width LaTeX places it at, so a figure wide enough to
    need a `figure*` is compared against :data:`TEXT_WIDTH` rather than one
    column.
    """
    scale = (figure_width / placed_width) * (COLUMN_WIDTH / CORNER_REFERENCE_WIDTH)
    return {name: size * scale
            for name, size in COLUMN_REFERENCE_FONT_SIZES.items()}


def column_font_rc(figure_width, placed_width=COLUMN_WIDTH):
    """rcParams to layer over PUBLICATION_RC for a figure this many inches wide.

    Text set explicitly on an artist is not covered; pass it
    :func:`column_font_sizes` instead.
    """
    sizes = column_font_sizes(figure_width, placed_width)
    return {
        "font.size": sizes["tick"],
        "axes.labelsize": sizes["label"],
        "axes.titlesize": sizes["title"],
        "xtick.labelsize": sizes["tick"],
        "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],
    }


def _deuterium_prior_samples(size, seed):
    """Draw the 2016 Deuterium kmax=8 result in its native z basis."""
    covariance = (np.outer(DEUTERIUM_K8_ERRORS, DEUTERIUM_K8_ERRORS)
                  * (DEUTERIUM_K8_CORRELATION
                     + DEUTERIUM_K8_CORRELATION.T) / 2)
    values, vectors = np.linalg.eigh(covariance)
    covariance = (vectors * np.clip(values, 0, None)) @ vectors.T
    draws = np.random.default_rng(seed).multivariate_normal(
        DEUTERIUM_K8_FREE, covariance, size=size
    )
    full = np.array([
        complete_zexp_a_values(
            draw, 8, -0.28, t_cut_gev2=9 * 0.139570**2,
            fa_q2_zero=AXIAL_FORM_FACTOR_Q2_ZERO,
        )
        for draw in draws
    ])
    return full, -0.28, 9 * 0.139570**2


def _reference_prior_samples(key, size, seed):
    """Return full coefficient samples and basis metadata for a named prior."""
    if key == "deuterium":
        return _deuterium_prior_samples(size, seed)
    if key == "deuterium_k6":
        free = np.random.default_rng(seed).multivariate_normal(
            DEUTERIUM_K6_FREE, DEUTERIUM_K6_COVARIANCE, size=size
        )
        central_completed = complete_zexp_a_values(
            DEUTERIUM_K6_FREE, 6, ZEXP_T0_GEV2,
            t_cut_gev2=ZEXP_T_CUT_GEV2, fa_q2_zero=ZEXP_FA_Q2_ZERO,
        )
        if not np.allclose(central_completed, DEUTERIUM_K6_FULL, atol=5e-8):
            raise ValueError(
                "The Deuterium kmax=6 coefficients do not satisfy the configured "
                "normalization and sum-rule convention"
            )
        full = np.array([
            complete_zexp_a_values(
                draw, 6, ZEXP_T0_GEV2, t_cut_gev2=ZEXP_T_CUT_GEV2,
                fa_q2_zero=ZEXP_FA_Q2_ZERO,
            )
            for draw in free
        ])
        return full, ZEXP_T0_GEV2, ZEXP_T_CUT_GEV2
    if key not in REFERENCE_PRIORS:
        raise KeyError(
            f"Unknown reference prior {key!r}; choose from "
            f"{sorted(('deuterium', 'deuterium_k6', *REFERENCE_PRIORS))}"
        )
    prior = REFERENCE_PRIORS[key]
    central, matrix = _zexp_transform(prior)
    pulls = np.random.default_rng(seed).normal(size=(size, matrix.shape[1]))
    return central + pulls @ matrix.T, prior.t0_gev2, prior.t_cut_gev2


def _fa_observables(coefficient_samples, t0_gev2, t_cut_gev2, q2_points):
    """Evaluate every coefficient sample at common physical Q2 points."""
    q2 = np.asarray(q2_points, dtype=float)
    if q2.ndim != 1 or len(q2) < 2 or not np.all(np.isfinite(q2)):
        raise ValueError("q2_points must contain at least two finite values")
    if np.any(q2 < 0):
        raise ValueError("q2_points must be non-negative")
    z = ((np.sqrt(t_cut_gev2 + q2) - np.sqrt(t_cut_gev2 - t0_gev2)) /
         (np.sqrt(t_cut_gev2 + q2) + np.sqrt(t_cut_gev2 - t0_gev2)))
    powers = z[None, :] ** np.arange(coefficient_samples.shape[1])[:, None]
    values = coefficient_samples @ powers
    if not np.all(np.isfinite(values)):
        raise ValueError("A selected distribution produced non-finite F_A values")
    return values


@mpl.rc_context(PUBLICATION_RC)
def plot_distribution_overlay(results, selections, bins=55,
                              n_reference_samples=50_000, seed=2026,
                              figsize=(7.0, 6.2), labels=None, colors=None,
                              smooth=1.0, legend_loc="upper center",
                              legend_headroom=.30, legend_clearance=.03,
                              note="68% and 95% credible regions"):
    """Overlay chosen priors/posteriors as contours in a common (a1, a2) basis.

    The 68% credible region of each distribution is filled and its 68% and
    95% boundaries are outlined; the posterior is solid and every prior takes
    its own pattern from ``PRIOR_DASH_PATTERNS``, so priors that nearly
    coincide can still be told apart where they overlap. Distributions are
    drawn from the broadest to the most compact. ``smooth`` is the Gaussian
    kernel width in bins, which a finer ``bins`` needs raised to keep the
    contours as smooth.

    Each selection is ``(key, distribution)`` where distribution is ``"prior"``
    or ``"posterior"``. Priors may use any loaded fit key or the standalone
    references ``deuterium``, ``deuterium_k6``, ``minerva_k6``, ``lqcd_k6``, and
    ``minerva_lqcd_k6``. Posteriors are read from ``results``. Every selected
    distribution must use the same t0 and t_cut convention; incompatible bases
    are rejected rather than silently overlaying unlike coefficients.

    Legend text follows :func:`prior_label` and :func:`posterior_label`:
    ``"<Source> prior, $k_max=N$"`` for priors, whether or not the fit was
    loaded into ``results``, and ``"Posterior, $k_max=N$"`` for posteriors,
    extended to ``"Posterior from <source> prior, ..."`` when the figure
    involves more than one prior. ``labels`` optionally overrides individual
    entries: key posteriors by fit key and priors by ``"<key> prior"``.

    Contours take their color from :data:`FA_SOURCE_COLORS`, so one fit key
    drawn as both a prior and a posterior would otherwise get that color
    twice; ``colors`` overrides individual entries, keyed exactly like
    ``labels``. ``note`` is the grey line above the axes, the place for
    whatever every distribution shares (a common k_max, say) instead of
    repeating it in every legend entry; pass ``""`` to drop it.

    The legend is split into a priors column and a posteriors column, and
    ``legend_headroom`` opens a band above the contours for it, as a fraction
    of the plotted y range. The band is added to the axes only, so the density
    estimate underneath is unchanged; set it to 0 when moving the legend to a
    corner the contours already leave free. The band is a minimum: wherever a
    contour still reaches under the legend's frame the top is raised further
    until the frame clears it by ``legend_clearance`` of the y range.
    """
    if not selections:
        raise ValueError("Select at least one prior or posterior distribution")
    labels = dict(labels or {})
    colors = dict(colors or {})
    fit_specs = {spec.key: spec for spec in SPECS if spec.prior is not None}
    distributions = []
    common_basis = None
    for index, (key, distribution) in enumerate(selections):
        spec = fit_specs.get(key)
        if distribution == "prior":
            if key in results and results[key]["spec"].prior is not None:
                result = results[key]
                coefficients = result["prior_samples"]
                prior = result["spec"].prior
                t0, t_cut = prior.t0_gev2, prior.t_cut_gev2
            else:
                coefficients, t0, t_cut = _reference_prior_samples(
                    key, n_reference_samples, seed + index
                )
            prior_text = REFERENCE_LABELS[key] if spec is None else prior_label(spec)
        elif distribution == "posterior":
            if key not in results:
                raise KeyError(
                    f"Posterior {key!r} is unavailable in this suite; run its fit "
                    "and the notebook's main results cell first"
                )
            result = results[key]
            if result["spec"].prior is None:
                raise ValueError(
                    f"{key!r} is a dipole-M_A fit, not a z-expansion distribution"
                )
            spec = result["spec"]
            coefficients = result["samples"]
            prior = spec.prior
            t0, t_cut = prior.t0_gev2, prior.t_cut_gev2
            prior_text = prior_label(spec)
        else:
            raise ValueError(
                f"Distribution for {key!r} must be 'prior' or 'posterior', "
                f"not {distribution!r}"
            )
        basis = (float(t0), float(t_cut))
        if common_basis is None:
            common_basis = basis
        elif not np.allclose(basis, common_basis, rtol=0, atol=1e-12):
            raise ValueError(
                f"{key!r} uses (t0, t_cut)={basis}, which is incompatible "
                f"with the selected common basis {common_basis}. Use the "
                "translated deuterium_k6 prior instead of deuterium."
            )
        if coefficients.ndim != 2 or coefficients.shape[1] < 3:
            raise ValueError(f"{key!r} does not contain physical a1 and a2 samples")
        values = coefficients[:, 1:3]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key!r} contains non-finite a1/a2 samples")
        color_key = f"{key} prior" if distribution == "prior" else key
        color = colors.get(
            color_key, FA_SOURCE_COLORS.get(key, f"C{index % 10}")
        )
        distributions.append((key, distribution, spec, prior_text, values, color))

    # With a single prior in the figure, posteriors need not name it.
    single_prior = len({item[3] for item in distributions}) == 1
    distributions = [
        (labels.get(f"{key} prior", prior_text) if distribution == "prior"
         else labels.get(key, posterior_label(spec, single_prior)),
         values, color, distribution)
        for key, distribution, spec, prior_text, values, color
        in distributions
    ]

    joined = np.concatenate([values for _, values, _, _ in distributions])
    xlow, ylow = np.quantile(joined, 0.001, axis=0)
    xhigh, yhigh = np.quantile(joined, 0.999, axis=0)
    xpad = .05 * (xhigh - xlow) if xhigh > xlow else .5
    ypad = .05 * (yhigh - ylow) if yhigh > ylow else .5
    plot_range = [(xlow - xpad, xhigh + xpad),
                  (ylow - ypad, yhigh + ypad)]

    # Print at the same text size as the corner figures beside it: this one is
    # wider at the source, so LaTeX shrinks it further (see column_font_sizes).
    font_sizes = column_font_sizes(figsize[0])
    with mpl.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Bin and smooth every distribution first: the drawing order is by area,
    # so a compact distribution is never buried under a broad one.
    styled = []
    prepared = []
    prior_index = 0
    for label, values, color, distribution in distributions:
        if distribution == "posterior":
            linestyle = "-"
        else:
            linestyle = PRIOR_DASH_PATTERNS[prior_index % len(PRIOR_DASH_PATTERNS)]
            prior_index += 1
        styled.append((label, color, linestyle, distribution))
        histogram, xedges, yedges = np.histogram2d(
            values[:, 0], values[:, 1], bins=bins, range=plot_range,
        )
        smoothed = _smooth_density_2d(histogram, smooth)
        levels = _credible_density_levels(smoothed)
        if not levels:
            continue
        prepared.append((
            int((smoothed >= levels[0]).sum()), color, linestyle, smoothed,
            xedges, yedges, levels,
        ))

    for order, (_, color, linestyle, smoothed, xedges, yedges, levels) in enumerate(
        sorted(prepared, key=lambda item: -item[0])
    ):
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        zorder = 2 + 2 * order
        # Fill the innermost (68%) region only. Filling both bands stacks four
        # translucent layers wherever distributions overlap and greys them out.
        ax.contourf(
            xcenters, ycenters, smoothed.T,
            levels=[levels[-1], smoothed.max() * 1.001],
            colors=[mpl.colors.to_rgba(color, CONTOUR_CORE_ALPHA)],
            zorder=zorder,
        )
        ax.contour(
            xcenters, ycenters, smoothed.T, levels=levels, colors=color,
            linewidths=np.linspace(*CONTOUR_LINEWIDTHS, len(levels)),
            linestyles=[linestyle] * len(levels), zorder=zorder + 1,
        )

    ax.set_xlim(plot_range[0])
    # The headroom is an axes limit, not a wider binning range, so the contours
    # are identical to the ones drawn without a legend band.
    ylow, yhigh = plot_range[1]
    ax.set_ylim(ylow, yhigh + legend_headroom * (yhigh - ylow))
    ax.set_xlabel(r"$a_1$", fontsize=font_sizes["label"])
    ax.set_ylabel(r"$a_2$", fontsize=font_sizes["label"])
    ax.tick_params(labelsize=font_sizes["tick"])
    ax.minorticks_on()
    ax.grid(color="#9AA4B2", alpha=.18, linewidth=.7, zorder=0)
    if note:
        ax.annotate(
            note, xy=(1.0, 1.015),
            xycoords="axes fraction", ha="right", va="bottom",
            color="#596273", fontsize=font_sizes["legend"],
        )

    # Swatches carry both the fill and the line style, so the legend reads the
    # same way the contours do.
    handles = {"prior": [], "posterior": []}
    for label, color, linestyle, distribution in styled:
        handles[distribution].append(
            Patch(facecolor=mpl.colors.to_rgba(color, CONTOUR_CORE_ALPHA + .05),
                  edgecolor=color, linewidth=1.5, linestyle=linestyle,
                  label=label)
        )
    # Priors in the left column, posteriors in the right. matplotlib fills a
    # legend column by column, so the shorter group is padded with blank
    # entries to keep the split exact whichever group is longer.
    columns = [column for column in
               (handles["prior"], handles["posterior"]) if column]
    rows = max(len(column) for column in columns)
    legend = ax.legend(
        handles=[handle for column in columns
                 for handle in column + [Line2D([], [], linestyle="none",
                                                label="")] * (rows - len(column))],
        ncol=len(columns), loc=legend_loc, fontsize=font_sizes["legend"],
        handlelength=2.1,
        handleheight=1.1, labelspacing=.7, handletextpad=.9, frameon=True,
        framealpha=.92, edgecolor="none", facecolor="white", borderpad=.8,
        borderaxespad=.8, columnspacing=1.8,
    )
    legend.set_zorder(20)

    # The legend is anchored in axes coordinates, so its lower edge sits at a
    # fixed axes fraction whatever the y range: raise the top until that
    # fraction lies above the highest contour vertex within the legend's x span.
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    axes_box = box.transformed(ax.transAxes.inverted())
    data_box = box.transformed(ax.transData.inverted())
    vertices = np.concatenate([
        path.vertices for collection in ax.collections
        for path in collection.get_paths()
    ])
    under = (vertices[:, 0] >= data_box.x0) & (vertices[:, 0] <= data_box.x1)
    if axes_box.y0 > 0 and under.any():
        ylow, ytop = ax.get_ylim()
        peak = vertices[under, 1].max() + legend_clearance * (ytop - ylow)
        ax.set_ylim(ylow, max(ytop, ylow + (peak - ylow) / axes_box.y0))
    return fig


# Default coordinates of the dipole-M_A overlay corner: M_A with the two
# unbounded knobs the data move most visibly (RPA correlates with M_A, FrCEx N
# is pulled by about one sigma). Any of a fit's ``ma_joint_names`` can be
# selected instead; the two-point switches draw ragged contours here because
# their posteriors are nearly flat, so they are better read off the grid.
MA_OVERLAY_PARAMETERS = ("M_A [GeV]", "RPA CCQE pull", "FrCEx N pull")


@mpl.rc_context(PUBLICATION_RC)
def plot_ma_posterior_overlay(results,
                              fit_keys=("ma", "ma_uniform"),
                              labels=None, parameters=MA_OVERLAY_PARAMETERS,
                              bins=45,
                              figsize=(CORNER_REFERENCE_WIDTH, CORNER_REFERENCE_WIDTH),
                              placed_width=COLUMN_WIDTH):
    """Overlay the joint posteriors of several dipole-M_A fits in one corner.

    ``parameters`` names the coordinates to show, as they appear in each
    fit's ``ma_joint_names`` (M_A and the floated knobs); the default is
    :data:`MA_OVERLAY_PARAMETERS`. Legend entries follow
    :func:`posterior_label` and name the prior whenever the selected fits use
    different priors; ``labels`` maps fit keys to replacement text. Fonts come
    from :func:`column_font_sizes` for a figure ``figsize[0]`` inches wide
    printed at ``placed_width``.
    """
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    parameters = tuple(parameters)
    overlays = []
    for index, key in enumerate(fit_keys):
        if key not in results:
            raise KeyError(
                f"Posterior {key!r} is unavailable; run that XML and the "
                "notebook's main results cell first"
            )
        result = results[key]
        if "ma_joint_samples" not in result:
            raise ValueError(f"{key!r} is not a dipole-M_A fit")
        names = list(result["ma_joint_names"])
        missing = [name for name in parameters if name not in names]
        if missing:
            raise ValueError(
                f"{key!r} has no parameter(s) {missing}; available: {names}"
            )
        columns = [names.index(name) for name in parameters]
        samples = np.asarray(result["ma_joint_samples"])[:, columns]
        overlays.append((key, result["spec"], samples,
                         colors[index % len(colors)], _smoothing_factor(result)))

    # Standard legend text; ``labels`` (keyed by fit key) overrides entries.
    labels = dict(labels or {})
    single_prior = len({prior_label(spec) for _, spec, _, _, _ in overlays}) == 1
    overlays = [
        (labels.get(key, posterior_label(spec, single_prior)), samples, color, smoothing)
        for key, spec, samples, color, smoothing in overlays
    ]

    n = len(parameters)
    bounds = [NUISANCE_BOUNDS.get(name) for name in parameters]
    joined = np.concatenate([samples for _, samples, _, _ in overlays], axis=0)
    ranges = []
    for coordinate in range(n):
        if bounds[coordinate] is not None:
            ranges.append(tuple(bounds[coordinate]))
            continue
        low, high = np.quantile(joined[:, coordinate], [0.001, 0.999])
        padding = .04 * (high - low) if high > low else .5
        ranges.append((low - padding, high + padding))

    fonts = column_font_sizes(figsize[0], placed_width)
    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row < col:
                ax.set_visible(False)
                continue
            # As in plot_corner: every diagonal panel after the first ends a
            # row of joint panels and is drawn sideways to line up with it.
            sideways = row == col and row > 0
            for label, samples, color, smoothing in overlays:
                if row == col:
                    centers, density = _smooth_density(
                        samples[:, col], bins, bounds=bounds[col],
                        smoothing=smoothing,
                    )
                    if sideways:
                        ax.plot(density, centers, color=color, lw=1.6)
                    else:
                        ax.plot(centers, density, color=color, lw=1.6)
                else:
                    x_range, x_bins = _panel_binning(
                        ranges[col], samples[:, col], bins, bounds=bounds[col])
                    y_range, y_bins = _panel_binning(
                        ranges[row], samples[:, row], bins, bounds=bounds[row])
                    histogram, xedges, yedges = np.histogram2d(
                        samples[:, col], samples[:, row],
                        bins=[x_bins, y_bins], range=[x_range, y_range],
                    )
                    smoothed = _smooth_density_2d(
                        histogram, sigma=smoothing,
                        reflect=(bounds[col] is not None, bounds[row] is not None),
                    )
                    levels = _credible_density_levels(smoothed)
                    if levels:
                        xgrid, ygrid, density = _density_grid_to_edges(
                            xedges, yedges, smoothed
                        )
                        ax.contour(
                            xgrid, ygrid, density.T, levels=levels,
                            colors=color,
                            linewidths=np.linspace(.9, 1.6, len(levels)),
                        )
            if sideways:
                ax.set_ylim(ranges[row])
                ax.set_xlim(left=0)
            else:
                ax.set_xlim(ranges[col])
            if row > col:
                ax.set_ylim(ranges[row])
            if row == col == 0:
                ax.set_yticks([])
            if row == n - 1 and not sideways:
                ax.set_xlabel(_parameter_label(parameters[col]),
                              fontsize=fonts["label"])
            elif not sideways:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(_parameter_label(parameters[row]),
                              fontsize=fonts["label"])
            elif col > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=fonts["tick"])
            if row == col == 0:
                _style_corner_ticks(ax, x_only=True, x_bounds=bounds[col])
            elif sideways:
                _style_corner_ticks(ax, y_bounds=bounds[row])
                ax.set_xticks([])
            else:
                _style_corner_ticks(ax, x_bounds=bounds[col], y_bounds=bounds[row])
            ax.grid(color="#9AA4B2", alpha=.15, linewidth=.6)

    handles = [Line2D([], [], color=color, lw=1.8, label=label)
               for label, _, color, _ in overlays]
    # Lay out the corner panels without the legend. An axes-attached legend
    # outside the first diagonal panel makes tight_layout reserve a large gap
    # between every column, dramatically shrinking the plots.
    fig.tight_layout(pad=.5)
    top_left = axes[0, 0].get_position()
    legend_x = (axes[0, 1].get_position().x0 + .01 if n > 1
                else top_left.x1 + .02)
    legend = fig.legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(legend_x, top_left.y1),
        fontsize=fonts["legend"], borderaxespad=0, handlelength=2.2,
    )
    legend.set_in_layout(False)
    return fig


# Shared drawing conventions of the corner-style figures: nested blue regions
# (darker 68%, lighter 95%, no outline), the prior in red, the profile best
# fit in black.
CORNER_POSTERIOR_COLOR = "#0072C1"
CORNER_68_FILL = "#7FB8DF"
CORNER_95_FILL = "#BFDBEF"
CORNER_BEST_FIT_COLOR = "#000000"
CORNER_PRIOR_COLOR = "#D62728"


def _corner_layout(n):
    """Source width and printed width of an n-parameter corner.

    Corners with up to four parameters are one-column figures drawn at the
    reference width; wider ones are drawn at the full text width and must be
    placed with ``figure*``. Fonts derive from :func:`column_font_sizes`, so
    every corner prints with the same type size.
    """
    if n <= 4:
        return CORNER_REFERENCE_WIDTH, COLUMN_WIDTH
    return TEXT_WIDTH, TEXT_WIDTH


def _display_ranges(names, samples, prior_samples, prior_mask, axis_ranges):
    """Per-parameter display range: hard bounds, then axis_ranges, then data."""
    ranges = []
    for i, name in enumerate(names):
        if name in NUISANCE_BOUNDS:
            ranges.append(tuple(NUISANCE_BOUNDS[name]))
            continue
        if axis_ranges is not None and name in axis_ranges:
            ranges.append(tuple(axis_ranges[name]))
            continue
        range_samples = (np.r_[samples[:, i], prior_samples[:, i]]
                         if prior_mask[i] else samples[:, i])
        low, high = range_samples.min(), range_samples.max()
        padding = .03 * (high - low) if high > low else .5
        ranges.append((low - padding, high + padding))
    return ranges


def _draw_marginal(ax, samples, prior_samples, best_fit, bins, bounds,
                   smoothing, show_prior, vertical=False):
    """Posterior (and prior) density line with the best-fit marker line."""
    centers, density = _smooth_density(samples, bins, bounds=bounds,
                                       smoothing=smoothing)
    curves = [(centers, density, CORNER_POSTERIOR_COLOR, 1.4, 1.0)]
    if show_prior:
        prior_centers, prior_density = _smooth_density(
            prior_samples, bins, bounds=bounds
        )
        curves.append((prior_centers, prior_density, CORNER_PRIOR_COLOR, 1.0, .6))
    for centers, density, color, lw, alpha in curves:
        if vertical:
            ax.plot(density, centers, color=color, lw=lw, alpha=alpha)
        else:
            ax.plot(centers, density, color=color, lw=lw, alpha=alpha)
    line = ax.axhline if vertical else ax.axvline
    line(best_fit, color=CORNER_BEST_FIT_COLOR, ls="--", lw=1.0)


def _draw_joint_panel(ax, x, y, x_range, y_range, x_bounds, y_bounds, bins,
                      smoothing, prior=None):
    """Nested 68%/95% posterior regions and optional prior contours.

    ``prior`` is ``(prior_x, prior_y)`` to draw the prior's 68% and 95%
    contours, or ``None``. Against a hard-bounded axis the prior is binned
    and reflected exactly like the posterior, so its contour reaches the
    bound (a truncated normal on [0, 1] then draws a box flush with it).
    """
    x_range, x_bins = _panel_binning(x_range, x, bins, bounds=x_bounds)
    y_range, y_bins = _panel_binning(y_range, y, bins, bounds=y_bounds)
    reflect = (x_bounds is not None, y_bounds is not None)
    histogram, xedges, yedges = np.histogram2d(
        x, y, bins=[x_bins, y_bins], range=[x_range, y_range],
    )
    histogram = _smooth_density_2d(histogram, sigma=smoothing, reflect=reflect)
    levels = _credible_density_levels(histogram)
    if levels:
        xgrid, ygrid, density = _density_grid_to_edges(xedges, yedges, histogram)
        upper = np.nextafter(density.max(), np.inf)
        if len(levels) == 2:
            ax.contourf(
                xgrid, ygrid, density.T, levels=[levels[0], levels[1], upper],
                colors=[CORNER_95_FILL, CORNER_68_FILL],
            )
        else:
            ax.contourf(
                xgrid, ygrid, density.T, levels=[levels[0], upper],
                colors=[CORNER_68_FILL],
            )
    if prior is not None:
        prior_x, prior_y = prior
        prior_x_range, prior_x_bins = _panel_binning(
            x_range, prior_x, bins, bounds=x_bounds)
        prior_y_range, prior_y_bins = _panel_binning(
            y_range, prior_y, bins, bounds=y_bounds)
        prior_histogram, prior_xedges, prior_yedges = np.histogram2d(
            prior_x, prior_y, bins=[prior_x_bins, prior_y_bins],
            range=[prior_x_range, prior_y_range],
        )
        prior_histogram = _smooth_density_2d(prior_histogram, reflect=reflect)
        prior_levels = _credible_density_levels(prior_histogram)
        if prior_levels:
            prior_xgrid, prior_ygrid, prior_density = _density_grid_to_edges(
                prior_xedges, prior_yedges, prior_histogram
            )
            ax.contour(
                prior_xgrid, prior_ygrid, prior_density.T, levels=prior_levels,
                colors=CORNER_PRIOR_COLOR,
                linewidths=np.linspace(.8, 1.1, len(prior_levels)),
                linestyles=["--", "-"][-len(prior_levels):], alpha=.6,
            )


def _draw_best_fit(ax, x, y):
    """Profile best fit in a joint panel: dashed crosshair lines and a marker.

    The lines continue the dashed best-fit line of the marginal panels, so
    the best fit can be read against every axis of every panel.
    """
    ax.axvline(x, color=CORNER_BEST_FIT_COLOR, ls="--", lw=.8, zorder=3)
    ax.axhline(y, color=CORNER_BEST_FIT_COLOR, ls="--", lw=.8, zorder=3)
    ax.plot(x, y, "D", color=CORNER_BEST_FIT_COLOR, markeredgecolor="white",
            markeredgewidth=.4, markersize=3.5, zorder=4)


def _corner_legend_handles(any_prior):
    handles = [
        Patch(facecolor=CORNER_68_FILL, label="Posterior 68%"),
        Patch(facecolor=CORNER_95_FILL, label="Posterior 95%"),
        Line2D([], [], color=CORNER_BEST_FIT_COLOR, marker="D", ls="none",
               markersize=4.5, label="Profile best fit"),
    ]
    if any_prior:
        handles.extend([
            Line2D([], [], color=CORNER_PRIOR_COLOR, lw=1.1, ls="-", alpha=.6,
                   label="Prior 68%"),
            Line2D([], [], color=CORNER_PRIOR_COLOR, lw=.9, ls="--", alpha=.6,
                   label="Prior 95%"),
        ])
    return handles


@mpl.rc_context(PUBLICATION_FONT_RC)
def plot_corner(result, bins=35, show_all_coefficients=False,
                show_prior=False, prior_mask=None, axis_names=None,
                axis_ranges=None, smoothing=None):
    """Joint physical posterior, restricted to independent coefficients by default.

    The full triangle: marginals on the diagonal (no titles; the intervals
    are in the fit's summary table; after the first they are drawn sideways
    so each row's best-fit line runs through its marginal, as in the
    nuisance grid), joint 68% and 95% regions below it, the
    prior overlaid where ``prior_mask`` allows (its contours reflected at
    hard bounds like the posterior), and the profile best fit as a dashed
    crosshair with a marker. Up to four parameters it is a
    one-column figure; beyond that it is drawn at the full text width for
    the supplement. :func:`plot_physical_nuisance_grid` is the compact form
    for the main text.

    ``axis_ranges`` optionally maps a parameter name (as in ``result["names"]``,
    e.g. ``"M_A [GeV]"`` or ``"a1"``) to a fixed ``(low, high)`` display range,
    so panels for that parameter share one x/y range across fits and suites.
    Parameters absent from the mapping keep their automatic, sample-derived
    range; hard-bounded knobs always show exactly their bounds. ``smoothing``
    overrides the kernel-width factor from :func:`_smoothing_factor`.
    """
    samples, names, indices = _diagnostic_view(result, show_all_coefficients)
    if axis_names is None:
        axis_names = names
    elif len(axis_names) != len(names):
        raise ValueError("axis_names must contain one label per plotted parameter")
    prior_samples = result["prior_samples"][:, indices]
    profile = result["profile"][indices]
    n = len(names)
    if prior_mask is None:
        prior_mask = np.full(n, bool(show_prior), dtype=bool)
    else:
        prior_mask = np.asarray(prior_mask, dtype=bool)
        if prior_mask.shape != (n,):
            raise ValueError(
                f"prior_mask must contain one value per plotted parameter ({n})"
            )
        prior_mask &= bool(show_prior)
    any_prior = bool(np.any(prior_mask))
    if smoothing is None:
        smoothing = _smoothing_factor(result)
    bounds = [NUISANCE_BOUNDS.get(name) for name in names]
    figure_width, placed_width = _corner_layout(n)
    fonts = column_font_sizes(figure_width, placed_width)
    fig, axes = plt.subplots(n, n, figsize=(figure_width, figure_width),
                             squeeze=False)
    ranges = _display_ranges(names, samples, prior_samples, prior_mask, axis_ranges)

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row < col:
                ax.set_visible(False)
                continue
            if row == col:
                # The first diagonal panel heads its column; every later one ends
                # its row of joint panels, so it is drawn sideways (parameter on
                # the y axis) and the row's dashed best-fit line runs straight
                # through it, as in plot_physical_nuisance_grid.
                sideways = row > 0
                _draw_marginal(
                    ax, samples[:, col], prior_samples[:, col], profile[col],
                    bins, bounds[col], smoothing, prior_mask[col],
                    vertical=sideways,
                )
                if sideways:
                    ax.set_ylim(ranges[row])
                    ax.set_xlim(left=0)
                    ax.set_xticks([])
                else:
                    ax.set_yticks([])
            else:
                prior = ((prior_samples[:, col], prior_samples[:, row])
                         if prior_mask[col] and prior_mask[row] else None)
                _draw_joint_panel(
                    ax, samples[:, col], samples[:, row], ranges[col], ranges[row],
                    bounds[col], bounds[row], bins, smoothing, prior,
                )
                _draw_best_fit(ax, profile[col], profile[row])
                ax.set_ylim(ranges[row])
            sideways_marginal = row == col and row > 0
            if not sideways_marginal:
                ax.set_xlim(ranges[col])
            if row == col == 0:
                _style_corner_ticks(ax, x_only=True, x_bounds=bounds[col])
            elif sideways_marginal:
                _style_corner_ticks(ax, y_bounds=bounds[row])
                ax.set_xticks([])
            else:
                _style_corner_ticks(ax, x_bounds=bounds[col], y_bounds=bounds[row])
            if row == n - 1 and not sideways_marginal:
                ax.set_xlabel(_parameter_label(axis_names[col]),
                              fontsize=fonts["label"])
            elif not sideways_marginal:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(_parameter_label(axis_names[row]),
                              fontsize=fonts["label"])
            elif col > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=fonts["tick"])

    # Lay out the corner grid first, then put the legend in the otherwise
    # unused upper triangle, beside the first diagonal panel. Keeping it out
    # of the layout prevents it from shrinking the panel beside it.
    fig.tight_layout(pad=.3)
    handles = _corner_legend_handles(any_prior)
    if n == 1:
        axes[0, 0].legend(handles=handles, loc="upper right",
                          fontsize=fonts["legend"], frameon=False,
                          labelspacing=.45, handlelength=1.8, handletextpad=.55)
        return fig
    top_diagonal = axes[0, 0].get_position()
    legend = fig.legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(axes[0, 1].get_position().x0 + .01, top_diagonal.y1),
        ncol=1, fontsize=fonts["legend"], frameon=False,
        labelspacing=.45, handlelength=1.8, handletextpad=.55,
        borderaxespad=0,
    )
    legend.set_in_layout(False)
    return fig


def _joint_view(result):
    """Physical parameters and floated knobs of one fit as one joint view.

    Returns ``(samples, prior_samples, profile, names, n_physical)``: the
    M_A fits contribute M_A, the ``*_nuisance`` z-expansion fits their free
    coefficients, each followed by the knobs in :data:`NUISANCE_LABELS` order.
    """
    if "ma_joint_samples" in result:
        return (result["ma_joint_samples"], result["ma_joint_prior_samples"],
                result["ma_joint_profile"], list(result["ma_joint_names"]), 1)
    if "zexp_nuisance_samples" in result:
        return (result["zexp_nuisance_samples"],
                result["zexp_nuisance_prior_samples"],
                result["zexp_nuisance_profile"],
                list(result["zexp_nuisance_names"]), len(result["joint_names"]))
    raise ValueError(f"{result['spec'].key!r} floats no modeling knobs")


def _joint_result(result):
    """``result`` re-pointed at its joint view, for :func:`plot_corner`."""
    samples, prior_samples, profile, names, _ = _joint_view(result)
    joint = dict(result)
    joint.update(
        samples=samples, prior_samples=prior_samples, profile=profile,
        names=names, joint_samples=samples, joint_names=names,
        joint_indices=np.arange(len(names)),
    )
    return joint


def plot_ma_nuisance_corner(result, bins=35, show_prior=False,
                            show_ma_prior=True, axis_ranges=None):
    """Full corner of physical M_A and its fitted cross-section nuisances."""
    n = len(result["ma_joint_names"])
    return plot_corner(
        _joint_result(result), bins=bins, show_prior=show_prior,
        prior_mask=[show_ma_prior] + [True] * (n - 1),
        axis_ranges=axis_ranges,
    )


def plot_zexp_nuisance_corner(result, bins=35, show_prior=False,
                              show_zexp_prior=True, axis_ranges=None):
    """Full corner of the physical z coefficients and fitted nuisances.

    ``show_zexp_prior=False`` suppresses the prior overlay on the z-expansion
    coefficients while keeping it on the nuisance parameters; use this for
    uniform-prior fits, whose synthetic ``prior_samples`` only exist to set
    axis ranges and do not represent a prior that entered the fit.
    """
    n = len(result["zexp_nuisance_names"])
    n_axial = len(result["joint_names"])
    return plot_corner(
        _joint_result(result), bins=bins, show_prior=show_prior,
        prior_mask=[show_zexp_prior] * n_axial + [True] * (n - n_axial),
        axis_ranges=axis_ranges,
    )


@mpl.rc_context(PUBLICATION_FONT_RC)
def plot_physical_nuisance_grid(result, bins=35, show_prior=True,
                                show_physical_prior=True, axis_ranges=None,
                                smoothing=None, figure_width=TEXT_WIDTH,
                                placed_width=TEXT_WIDTH):
    """Compact joint view: the physical parameters against every floated knob.

    Rows are the physical parameters (M_A, or the free z-expansion
    coefficients), columns the knobs. The top row holds each knob's marginal
    posterior with its prior, the right column the physical marginals, and
    the inner panels the joint 68% and 95% credible regions with the profile
    best fit (dashed crosshair and marker). Knob-knob panels are left out:
    their posterior correlations stay below 0.15 in every fit, so they only
    repeat the prior;
    :func:`plot_posterior_correlation` gives the numbers. This is the
    main-text figure, drawn at the full text width (``figure*``);
    :func:`plot_corner` keeps the full triangle for the supplement.

    ``show_physical_prior=False`` drops the prior from the physical marginals
    and from the joint panels (uniform-prior fits). ``axis_ranges`` and
    ``smoothing`` behave as in :func:`plot_corner`.
    """
    samples, prior_samples, profile, names, n_physical = _joint_view(result)
    n_knobs = len(names) - n_physical
    if n_knobs == 0:
        raise ValueError(f"{result['spec'].key!r} floats no modeling knobs")
    if smoothing is None:
        smoothing = _smoothing_factor(result)
    physical = list(range(n_physical))
    knobs = list(range(n_physical, len(names)))
    bounds = [NUISANCE_BOUNDS.get(name) for name in names]
    prior_mask = np.array(
        [show_prior and show_physical_prior] * n_physical
        + [show_prior] * n_knobs
    )
    ranges = _display_ranges(names, samples, prior_samples, prior_mask, axis_ranges)

    fonts = column_font_sizes(figure_width, placed_width)
    # The marginal strips are shorter than the joint panels; the right column
    # is a little wider than the top row is tall so that the legend fits in
    # the unused top-right slot.
    marginal_height, marginal_width = .6, .9
    n_rows, n_cols = n_physical + 1, n_knobs + 1
    panel = figure_width / (n_knobs + marginal_width)
    figure_height = panel * (n_physical + marginal_height) + .35
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figure_width, figure_height), squeeze=False,
        gridspec_kw={
            "height_ratios": [marginal_height] + [1] * n_physical,
            "width_ratios": [1] * n_knobs + [marginal_width],
        },
    )

    for j, k in enumerate(knobs):
        ax = axes[0, j]
        _draw_marginal(
            ax, samples[:, k], prior_samples[:, k], profile[k], bins,
            bounds[k], smoothing, prior_mask[k],
        )
        ax.set_xlim(ranges[k])
        ax.set_yticks([])
        _style_corner_ticks(ax, x_only=True, x_bounds=bounds[k])
        ax.set_xticklabels([])
    axes[0, n_knobs].set_visible(False)

    for i, p in enumerate(physical):
        for j, k in enumerate(knobs):
            ax = axes[i + 1, j]
            prior = ((prior_samples[:, k], prior_samples[:, p])
                     if prior_mask[k] and prior_mask[p] else None)
            _draw_joint_panel(
                ax, samples[:, k], samples[:, p], ranges[k], ranges[p],
                bounds[k], bounds[p], bins, smoothing, prior,
            )
            _draw_best_fit(ax, profile[k], profile[p])
            ax.set_xlim(ranges[k])
            ax.set_ylim(ranges[p])
            _style_corner_ticks(ax, x_bounds=bounds[k], y_bounds=bounds[p])
            if i == n_physical - 1:
                ax.set_xlabel(_parameter_label(names[k]), fontsize=fonts["label"])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(_parameter_label(names[p]), fontsize=fonts["label"])
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=fonts["tick"])
        ax = axes[i + 1, n_knobs]
        _draw_marginal(
            ax, samples[:, p], prior_samples[:, p], profile[p], bins,
            bounds[p], smoothing, prior_mask[p], vertical=True,
        )
        ax.set_ylim(ranges[p])
        ax.set_xlim(left=0)
        _style_corner_ticks(ax, y_bounds=bounds[p])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.tick_params(labelsize=fonts["tick"])

    # Lay out the panels first, then put the legend in the unused top-right
    # slot so it does not shrink the panels around it.
    fig.tight_layout(pad=.3)
    slot = axes[0, n_knobs].get_position()
    fig.legend(
        handles=_corner_legend_handles(bool(np.any(prior_mask))),
        loc="upper right", bbox_to_anchor=(slot.x1, slot.y1),
        ncol=1, fontsize=fonts["legend"], frameon=False,
        labelspacing=.4, handlelength=1.3, handletextpad=.5, borderaxespad=0,
    )
    return fig


@mpl.rc_context(PUBLICATION_FONT_RC)
def plot_posterior_correlation(result, figure_width=COLUMN_WIDTH,
                               placed_width=COLUMN_WIDTH):
    """Posterior correlation matrix of the joint view, one number per pair.

    Replaces the knob-knob panels of the full corner. Only the strictly lower
    triangle is drawn (rows from the second parameter, columns to the
    second-to-last), physical parameters first, on a diverging
    red-white-blue scale.
    """
    samples, _, _, names, _ = _joint_view(result)
    correlation = np.atleast_2d(np.corrcoef(samples, rowvar=False))
    n = len(names)
    if n < 2:
        raise ValueError("a correlation matrix needs at least two parameters")
    # Row i of the drawing is parameter i + 1, column j is parameter j.
    lower = correlation[1:, :-1]
    hidden = np.triu(np.ones((n - 1, n - 1), dtype=bool), k=1)
    shown = np.ma.array(lower, mask=hidden)
    fonts = column_font_sizes(figure_width, placed_width)
    fig, ax = plt.subplots(figsize=(figure_width, figure_width * .9))
    colormap = mpl.colormaps["RdBu_r"].copy()
    colormap.set_bad("white")
    image = ax.imshow(shown, cmap=colormap, vmin=-1, vmax=1)
    for i in range(n - 1):
        for j in range(i + 1):
            value = lower[i, j]
            if abs(value) < .005:
                value = 0.0
            ax.text(j, i, f"{value:.2f}".replace("0.", "."),
                    ha="center", va="center", fontsize=fonts["tick"],
                    color="white" if abs(value) > .6 else "black")
    labels = [_parameter_label(name) for name in names]
    ax.set_xticks(range(n - 1), labels[:-1], rotation=45, ha="right",
                  rotation_mode="anchor", fontsize=fonts["tick"])
    ax.set_yticks(range(n - 1), labels[1:], fontsize=fonts["tick"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    colorbar.set_label("Posterior correlation", fontsize=fonts["label"])
    colorbar.ax.tick_params(labelsize=fonts["tick"])
    colorbar.outline.set_visible(False)
    fig.tight_layout(pad=.3)
    return fig


def plot_fit(result, bins=45, axis_ranges=None):
    """One compact figure: marginal distributions above, intervals below.

    ``axis_ranges`` optionally maps a parameter name (as in ``result["names"]``)
    to a fixed ``(low, high)`` x-axis range, so the same parameter shares one
    range across fits and suites. Parameters absent from the mapping keep
    their automatic range. Pass ``None`` (the default) to fix no range.
    """
    n = len(result["names"])
    fig, axes = plt.subplots(2, n, figsize=(max(6, 2.05 * n), 5.1),
                             squeeze=False, height_ratios=(2.2, 1),
                             sharex="col")
    for i, name in enumerate(result["names"]):
        top, bottom = axes[:, i]
        edges = np.histogram_bin_edges(
            np.r_[result["prior_samples"][:, i], result["samples"][:, i]], bins=bins
        )
        top.hist(result["prior_samples"][:, i], edges, density=True,
                 histtype="step", color="C3", lw=1.4, label="Prior")
        top.hist(result["samples"][:, i], edges, density=True,
                 histtype="stepfilled", color="C0", alpha=.4, label="Posterior")
        top.axvline(result["profile"][i], color="k", ls="--", label="Best fit")
        top.axvspan(result["q16"][i], result["q84"][i], color="C1", alpha=.18,
                    label="68% interval")
        top.set_title(_parameter_label(name))
        top.set_yticks([])
        if axis_ranges is not None and name in axis_ranges:
            top.set_xlim(axis_ranges[name])

        bottom.errorbar(result["central"][i], 1, xerr=result["sigma"][i],
                        fmt="o", color="C3", capsize=2)
        lo, hi = result["q16"][i], result["q84"][i]
        bottom.hlines(0, lo, hi, color="C0", linewidth=2)
        bottom.plot(result["profile"][i], 0, "D", color="C0", markersize=5)
        bottom.set_yticks([0, 1], ["Posterior", "Prior"] if i == 0 else ["", ""])
        bottom.grid(axis="x", alpha=.25)
        bottom.set_ylim(-.6, 1.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, ncol=1, loc="center right",
        bbox_to_anchor=(0.995, 0.53), fontsize=11,
        frameon=True, fancybox=True, framealpha=1.0,
        borderpad=0.8, labelspacing=0.9, handlelength=2.4,
        handletextpad=0.8,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.75")
    fig.suptitle(f'{result["spec"].title} — physical-parameter posterior', y=.995)
    fig.tight_layout(rect=(0, 0, .84, .94))
    return fig


def _save_figure(fig, suite, fit, stem, output_dir, dpi, formats):
    """Save one figure below ``<output_dir>/<suite>/<fit>/``."""
    fit_dir = Path(output_dir) / suite / fit
    fit_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(formats, str):
        formats = (formats,)
    saved = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        path = fit_dir / f"{stem}.{extension}"
        kwargs = {
            "bbox_inches": "tight", "pad_inches": .02, "facecolor": "white",
        }
        if extension not in {"pdf", "svg", "eps"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    return saved


def _fa_curves(result, q2, use_prior=False, max_samples=20_000, seed=2026):
    """Evaluate F_A for a representative subset of a fit's joint samples."""
    samples = result["prior_samples" if use_prior else "samples"]
    if len(samples) > max_samples:
        indices = np.random.default_rng(seed).choice(
            len(samples), size=max_samples, replace=False
        )
        samples = samples[indices]
    if result["spec"].prior is None:
        # GENIE convention used throughout this analysis: F_A(0) < 0.
        curves = -1.2723 / (1 + q2[None, :] / samples[:, :1] ** 2) ** 2
    else:
        prior = result["spec"].prior
        z = (
            np.sqrt(prior.t_cut_gev2 + q2)
            - np.sqrt(prior.t_cut_gev2 - prior.t0_gev2)
        ) / (
            np.sqrt(prior.t_cut_gev2 + q2)
            + np.sqrt(prior.t_cut_gev2 - prior.t0_gev2)
        )
        curves = samples @ np.vander(z, N=samples.shape[1], increasing=True).T
    return curves


@mpl.rc_context(PUBLICATION_RC)
def plot_fa_summary(results, show=None, comparison_prior=None,
                    q2_range=(0.0, 2.0), n_q2=401, max_samples=20_000,
                    ratio_zoom=None, xscale="linear", labels=None):
    """Publication figure of selected posteriors and one comparison prior.

    Parameters
    ----------
    results : mapping
        Output of :func:`run_suite`.
    show : iterable of str, optional
        Fit keys whose posterior F_A distributions are drawn.  ``None`` draws
        every available fit; an empty iterable draws none.
    comparison_prior : str, optional
        Fit key whose *prior* is drawn and used as the ratio denominator.
        This is independent of ``show`` and need not be a displayed posterior.
    ratio_zoom : tuple, optional
        ``(q2_min, q2_max, ratio_min, ratio_max)`` for a dedicated zoom panel.
        Use ``None`` to retain the simpler two-panel figure.
    xscale : {"linear", "log"}
        Scale shared by every Q^2 axis. Log scale requires positive limits.
    labels : mapping, optional
        Legend overrides: posteriors keyed by fit key, the comparison prior by
        ``"<key> prior"``. Unlisted entries use :func:`prior_label` and
        :func:`posterior_label`; posteriors name their prior only when the
        figure involves more than one prior.
    """
    available = tuple(results)
    selected = available if show is None else tuple(show)
    unknown = set(selected) - set(available)
    if unknown:
        raise KeyError(f"Unknown posterior key(s): {sorted(unknown)}; available: {available}")
    if comparison_prior is not None and comparison_prior not in results:
        raise KeyError(
            f"Unknown comparison prior {comparison_prior!r}; available: {available}"
        )
    if not selected and comparison_prior is None:
        raise ValueError("Select at least one posterior or a comparison prior")
    labels = dict(labels or {})
    prior_texts = {prior_label(results[key]["spec"]) for key in selected}
    if comparison_prior is not None:
        prior_texts.add(prior_label(results[comparison_prior]["spec"]))
    single_prior = len(prior_texts) <= 1

    if xscale not in {"linear", "log"}:
        raise ValueError("xscale must be 'linear' or 'log'")
    if xscale == "log" and (q2_range[0] <= 0 or
                             (ratio_zoom is not None and ratio_zoom[0] <= 0)):
        raise ValueError("Q2 lower limits must be positive when xscale='log'")
    q2 = (np.geomspace(*q2_range, n_q2) if xscale == "log"
          else np.linspace(*q2_range, n_q2))
    prior_quantiles = None
    if comparison_prior is not None:
        prior_curves = _fa_curves(
            results[comparison_prior], q2, use_prior=True,
            max_samples=max_samples,
        )
        prior_quantiles = np.quantile(prior_curves, [.16, .50, .84], axis=0)
        # Plot the conventional negative axial form factor as the positive
        # quantity -F_A; reverse the bounds when negating the interval.
        prior_quantiles = -prior_quantiles[::-1]
        denominator = prior_quantiles[1]
    else:
        denominator = 1.2723 / (1 + q2 / 1.014**2) ** 2

    # Colorblind-safe, print-friendly palette. Distinguish curves with both
    # hue and line style so the figure remains readable in grayscale.
    linestyles = ("-", "--", "-.", ":")
    legend_handles, legend_labels = [], []
    if ratio_zoom is None:
        fig, (ax, ratio) = plt.subplots(
            2, 1, figsize=(7.1, 6.4), sharex=True,
            gridspec_kw={"height_ratios": (3.15, 1), "hspace": .06},
        )
        zoom_axis = None
    else:
        if len(ratio_zoom) != 4:
            raise ValueError("ratio_zoom must be (q2_min, q2_max, ratio_min, ratio_max)")
        fig = plt.figure(figsize=(7.1, 6.4))
        grid = fig.add_gridspec(
            2, 3, height_ratios=(3.15, 1), width_ratios=(1, 1, 1.08),
            hspace=.30, wspace=.34,
        )
        ax = fig.add_subplot(grid[0, :])
        ratio = fig.add_subplot(grid[1, :2])
        zoom_axis = fig.add_subplot(grid[1, 2])

    if prior_quantiles is not None:
        label = labels.get(
            f"{comparison_prior} prior",
            prior_label(results[comparison_prior]["spec"]),
        )
        # Keep the common pre-fit reference neutral. Supplying alpha through
        # the face RGBA (rather than Collection.alpha) leaves hatch strokes
        # opaque and therefore visible in vector PDF output.
        prior_color = "#303030"
        prior_face = to_rgba("#B8B8B8", .25)
        ax.fill_between(q2, prior_quantiles[0], prior_quantiles[2],
                        facecolor=prior_face, edgecolor=prior_color,
                        linewidth=.35, hatch="////", zorder=0)
        ax.plot(q2, prior_quantiles[1], color=prior_color, lw=1.5,
                zorder=1)
        legend_handles.append((
            Patch(facecolor=prior_face, edgecolor=prior_color,
                  hatch="////", linewidth=.35),
            Line2D([], [], color=prior_color, lw=1.5),
        ))
        legend_labels.append(label)
        ratio.fill_between(
            q2, prior_quantiles[0] / denominator,
            prior_quantiles[2] / denominator,
            facecolor=prior_face, edgecolor=prior_color,
            linewidth=.35, hatch="////", zorder=0,
        )
        if zoom_axis is not None:
            zoom_axis.fill_between(
                q2, prior_quantiles[0] / denominator,
                prior_quantiles[2] / denominator,
                facecolor=prior_face, edgecolor=prior_color,
                linewidth=.35, hatch="////", zorder=0,
            )

    for i, key in enumerate(selected):
        result = results[key]
        curves = _fa_curves(result, q2, max_samples=max_samples)
        low, median, high = _quantile(curves, [.16, .50, .84])
        low, median, high = -high, -median, -low
        color = FA_SOURCE_COLORS.get(key, f"C{i % 10}")
        linestyle = linestyles[(i // 8) % len(linestyles)]
        ax.fill_between(q2, low, high, color=color, alpha=.25, linewidth=0)
        ax.plot(q2, median, color=color, ls=linestyle, lw=2,
                )
        legend_handles.append((
            Patch(facecolor=to_rgba(color, .25), edgecolor="none"),
            Line2D([], [], color=color, ls=linestyle, lw=2),
        ))
        legend_labels.append(
            labels.get(key, posterior_label(result["spec"], single_prior))
        )
        ratio.fill_between(q2, low / denominator, high / denominator,
                           color=color, alpha=.25, linewidth=0)
        ratio.plot(q2, median / denominator, color=color, ls=linestyle, lw=2)
        if zoom_axis is not None:
            zoom_axis.fill_between(q2, low / denominator, high / denominator,
                                   color=color, alpha=.25, linewidth=0)
            zoom_axis.plot(q2, median / denominator, color=color,
                           ls=linestyle, lw=2)

    ratio.axhline(1, color="0.35", lw=1, ls=(0, (2, 2)), zorder=-1)
    ax.set_ylabel(r"$-F_A(Q^2)$")
    ax.set_xlabel(r"$Q^2$ [GeV$^2$]")
    ratio.set_ylabel("Posterior / prior")
    ratio.set_xlabel(r"$Q^2$ [GeV$^2$]")
    ratio.set_xlim(q2_range)
    ratio.set_xscale(xscale)
    ax.set_xscale(xscale)
    if xscale == "log":
        # Match the publication-prior figure's fixed Q^2 ticks and labels.
        publication_ticks = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0)
        publication_labels = ("0.01", "0.05", "0.1", "0.5", "1", "2")
        visible = [
            (tick, label) for tick, label in
            zip(publication_ticks, publication_labels)
            if q2_range[0] <= tick <= q2_range[1]
        ]
        ratio.set_xticks(
            [tick for tick, _ in visible],
            labels=[label for _, label in visible],
        )
        ax.set_xticks(
            [tick for tick, _ in visible],
            labels=[label for _, label in visible],
        )
        ax.tick_params(axis="x", labelbottom=True)
    if zoom_axis is not None:
        q2_low, q2_high, ratio_low, ratio_high = ratio_zoom
        zoom_axis.axhline(1, color="0.35", lw=1,
                          ls=(0, (2, 2)), zorder=-1)
        zoom_axis.set_xlim(q2_low, q2_high)
        zoom_axis.set_ylim(ratio_low, ratio_high)
        zoom_axis.set_xscale(xscale)
        if xscale == "log":
            zoom_visible = [
                (tick, label) for tick, label in
                zip(publication_ticks, publication_labels)
                if q2_low <= tick <= q2_high
            ]
            zoom_axis.set_xticks(
                [tick for tick, _ in zoom_visible],
                labels=[label for _, label in zoom_visible],
            )
        zoom_axis.set_xlabel(r"$Q^2$ [GeV$^2$]")
        zoom_axis.set_title("Low-$Q^2$ detail", fontsize=9, pad=4)
        zoom_axis.tick_params(labelsize=8)
    styled_axes = (ax, ratio) if zoom_axis is None else (ax, ratio, zoom_axis)
    for axis in styled_axes:
        axis.grid(which="major", color="#9AA4B2", alpha=.22, linewidth=.7)
        axis.grid(which="minor", axis="x", color="#9AA4B2",
                  alpha=.10, linewidth=.5)
        axis.tick_params(direction="in", top=True, right=True)
    ax.legend(
        legend_handles, legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        frameon=False, fontsize=8.5, ncol=1, loc="lower left",
        handlelength=2.8, columnspacing=1.2,
    )
    fig.align_ylabels()
    return fig


def run_suite(suite, burn_in=0, thin=1, n_prior=50_000,
              show_all_coefficients=False, show_prior_in_corner=False,
              save_figures=True, output_dir=FIGURE_ROOT, save_dpi=600,
              save_formats=("pdf",), axis_ranges=None):
    """Load, summarize, plot, and optionally save every measurement in a suite.

    ``axis_ranges`` optionally maps a parameter name (e.g. ``"M_A [GeV]"`` or
    ``"a1"``) to a fixed ``(low, high)`` display range, applied to every
    marginal and corner panel for that parameter across every fit in the
    suite. This is how parameters that appear in multiple fits (an M_A fit
    across suites, a shared z-expansion coefficient across priors) end up
    with directly comparable axes. Parameters absent from the mapping keep
    their automatic, per-fit range; pass ``None`` (the default) to fix no
    parameter's range.
    """
    results = {}
    for spec in SPECS:
        result = load_fit(spec, suite, burn_in, thin, n_prior)
        if result is None:
            print(f"Skipping {spec.key}: no unique PROfile ROOT file found")
            continue
        results[spec.key] = result
        print(f'\n{spec.title}\n{result["root_file"]}')
        n_retained = len(result["samples"])
        print(f"Retained posterior samples after burn-in/thinning: {n_retained:,}")
        times = result["autocorrelation_times"]
        slowest = max(times, key=times.get)
        print(f"Integrated autocorrelation time: {times[slowest]:.1f} steps "
              f"({slowest}); effective samples: {result['ess_min']:,.0f}")
        if result["ess_min"] < SMOOTHING_REFERENCE_ESS:
            print(f"WARNING: fewer than {SMOOTHING_REFERENCE_ESS:,.0f} effective "
                  "samples; the densities are drawn with a wider kernel. Run a "
                  "longer chain (MCMC_ITERATIONS) rather than thinning.")
        if ("AxFFCCQEshape_UBGenie" in spec.chain_branches
                and np.any(result["ma_joint_samples"][:, 1] < 0)):
            print("WARNING: this ROOT chain contains AxFFCCQEshape values below 0 "
                  "and predates the XML restrict=\"0, 1\" setting; rerun the fit.")
        display(result["summary"].round(5))
        diagnostic_samples, diagnostic_names, _ = _diagnostic_view(
            result, show_all_coefficients
        )
        covariance_matrix = np.atleast_2d(np.cov(diagnostic_samples, rowvar=False))
        covariance = pd.DataFrame(
            covariance_matrix, index=diagnostic_names, columns=diagnostic_names,
        )
        correlation = pd.DataFrame(
            _correlation_from_covariance(covariance_matrix),
            index=diagnostic_names, columns=diagnostic_names,
        )
        coefficient_label = "All" if show_all_coefficients else "Free"
        print(f"{coefficient_label} physical-coefficient posterior covariance")
        display(covariance.round(5))
        print(f"{coefficient_label} physical-coefficient posterior correlation")
        display(correlation.round(3))
        fit_figure = plot_fit(result, axis_ranges=axis_ranges)
        if save_figures:
            paths = _save_figure(
                fit_figure, suite, spec.key, "physical_parameter_marginals",
                output_dir, save_dpi, save_formats,
            )
            print("Saved:", ", ".join(str(path) for path in paths))
        display(fit_figure)
        plt.close()
        corner_figure = plot_corner(
            result, show_all_coefficients=show_all_coefficients,
            show_prior=show_prior_in_corner and not spec.uniform_prior,
            axis_ranges=axis_ranges,
        )
        if save_figures:
            paths = _save_figure(
                corner_figure, suite, spec.key, "physical_parameter_corner",
                output_dir, save_dpi, save_formats,
            )
            print("Saved:", ", ".join(str(path) for path in paths))
        display(corner_figure)
        plt.close()
        if spec.prior is None:
            nuisance_figure = plot_ma_nuisance_corner(
                result,
                show_prior=show_prior_in_corner,
                show_ma_prior=not spec.uniform_prior,
                axis_ranges=axis_ranges,
            )
            if save_figures:
                paths = _save_figure(
                    nuisance_figure, suite, spec.key, "ma_nuisance_corner",
                    output_dir, save_dpi, save_formats,
                )
                print("Saved:", ", ".join(str(path) for path in paths))
            display(nuisance_figure)
            plt.close()
        elif spec.nuisance_branches:
            nuisance_figure = plot_zexp_nuisance_corner(
                result, show_prior=show_prior_in_corner,
                show_zexp_prior=not spec.uniform_prior,
                axis_ranges=axis_ranges,
            )
            if save_figures:
                paths = _save_figure(
                    nuisance_figure, suite, spec.key,
                    "zexp_nuisance_corner", output_dir, save_dpi, save_formats,
                )
                print("Saved:", ", ".join(str(path) for path in paths))
            display(nuisance_figure)
            plt.close()
        if spec.prior is None or spec.nuisance_branches:
            grid_figure = plot_physical_nuisance_grid(
                result, show_prior=show_prior_in_corner,
                show_physical_prior=not spec.uniform_prior,
                axis_ranges=axis_ranges,
            )
            if save_figures:
                paths = _save_figure(
                    grid_figure, suite, spec.key, "physical_nuisance_grid",
                    output_dir, save_dpi, save_formats,
                )
                print("Saved:", ", ".join(str(path) for path in paths))
            display(grid_figure)
            plt.close()
            correlation_figure = plot_posterior_correlation(result)
            if save_figures:
                paths = _save_figure(
                    correlation_figure, suite, spec.key, "posterior_correlation",
                    output_dir, save_dpi, save_formats,
                )
                print("Saved:", ", ".join(str(path) for path in paths))
            display(correlation_figure)
            plt.close()
    return results
