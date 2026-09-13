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
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
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
    "ma_no_axff": "#0072B2",
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


SPECS = (
    FitSpec("ma", r"Dipole $M_A$", source=r"Gaussian $M_A$",
            profile_labels=("MACCQE", "AxFFCCQEshape", "NormCCMEC", "RPA_CCQE"),
            chain_branches=("MaCCQE_UBGenie", "AxFFCCQEshape_UBGenie",
                            "NormCCMEC_UBGenie", "RPA_CCQE_UBGenie")),
    FitSpec("ma_no_axff", r"Dipole $M_A$ without AxFFCCQEshape",
            source=r"Gaussian $M_A$", variant="without AxFFCCQEshape",
            profile_labels=("MACCQE", "NormCCMEC", "RPA_CCQE"),
            chain_branches=("MaCCQE_UBGenie", "NormCCMEC_UBGenie",
                            "RPA_CCQE_UBGenie")),
    FitSpec("ma_uniform", r"Dipole $M_A$ without an $M_A$ pull penalty",
            source=r"uniform $M_A$",
            profile_labels=("MACCQE", "NormCCMEC", "RPA_CCQE"),
            chain_branches=("MaCCQE_UBGenie", "NormCCMEC_UBGenie",
                            "RPA_CCQE_UBGenie"),
            uniform_prior=True),
    FitSpec("lqcd_k6", r"LQCD (2026), $k_{\max}=6$", LQCD_K6_PRIOR,
            source="LQCD (2026)"),
    FitSpec(
        "lqcd_k6_nuisance",
        r"LQCD (2026), $k_{\max}=6$, fitted nuisances",
        LQCD_K6_PRIOR,
        source="LQCD (2026)", variant="fitted nuisances",
        nuisance_labels=("NormCCMEC", "RPA_CCQE"),
        nuisance_branches=("NormCCMEC_UBGenie", "RPA_CCQE_UBGenie"),
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
        nuisance_labels=("NormCCMEC", "RPA_CCQE"),
        nuisance_branches=("NormCCMEC_UBGenie", "RPA_CCQE_UBGenie"),
    ),
    FitSpec("minerva_k6_uniform",
            r"$k_{\max}=6$ uniform prior",
            MINERVA_K6_PRIOR, uniform_prior=True, source="uniform"),
    FitSpec(
        "minerva_k6_uniform_nuisance",
        r"$k_{\max}=6$ uniform prior, fitted nuisances",
        MINERVA_K6_PRIOR, uniform_prior=True,
        source="uniform", variant="fitted nuisances",
        nuisance_labels=("NormCCMEC", "RPA_CCQE"),
        nuisance_branches=("NormCCMEC_UBGenie", "RPA_CCQE_UBGenie"),
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
        nuisance_labels=("NormCCMEC", "RPA_CCQE"),
        nuisance_branches=("NormCCMEC_UBGenie", "RPA_CCQE_UBGenie"),
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
        nuisance_names = {
            "AxFFCCQEshape": "AxFFCCQEshape pull",
            "NormCCMEC": "NormCCMEC pull",
            "RPA_CCQE": "RPA CCQE pull",
        }
        nuisance_prior_ranges = {
            "AxFFCCQEshape": (0, 1),
            "NormCCMEC": (-2, 3),
            "RPA_CCQE": (-3, 3),
        }
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
    result = dict(spec=spec, root_file=root_file, central=central, sigma=sigma,
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
        nuisance_names = {
            "NormCCMEC": "NormCCMEC pull",
            "RPA_CCQE": "RPA CCQE pull",
        }
        nuisance_prior_ranges = {
            "NormCCMEC": (-2, 3),
            "RPA_CCQE": (-3, 3),
        }
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


def _smooth_density(samples, bins):
    """Return a lightly smoothed 1D histogram density for line display."""
    density, edges = np.histogram(samples, bins=max(80, 2 * bins), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    radius, sigma = 5, 1.5
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    # Divide by the kernel's overlap with the histogram. Without this the
    # convolution averages in the zeros beyond the first and last bin, which
    # drags the curve towards zero exactly where a posterior piles up against
    # a hard parameter bound.
    coverage = np.convolve(np.ones_like(density), kernel, mode="same")
    return centers, np.convolve(density, kernel, mode="same") / coverage


def _smooth_density_2d(histogram, sigma=1.0):
    """Apply a small separable Gaussian kernel to a 2D density estimate.

    ``sigma`` is in bins, so a finer binning needs a larger value to hold the
    physical smoothing length fixed. The default reproduces the kernel every
    caller used before the parameter existed.
    """
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.apply_along_axis(
        lambda values: np.convolve(values, kernel, mode="same"), 0, histogram
    )
    return np.apply_along_axis(
        lambda values: np.convolve(values, kernel, mode="same"), 1, smoothed
    )


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


def _panel_binning(display_range, values, bins, max_growth=4):
    """Histogram bounds and bin count for one axis of a corner panel.

    The bounds cover the panel, or the contours stop inside it wherever
    ``axis_ranges`` asks for a wider view than the chain reaches, and they
    cover every sample, because discarding samples would renormalize the
    credible levels. The bin count grows with the widened bounds so that the
    bin width stays the one ``bins`` would have given over the samples alone,
    which keeps the contour resolution independent of the requested view.
    """
    low = min(display_range[0], float(values.min()))
    high = max(display_range[1], float(values.max()))
    span = float(values.max() - values.min())
    if span <= 0 or high <= low:
        return (low, high), bins
    widened = int(np.ceil(bins * (high - low) / span))
    return (low, high), int(min(widened, max_growth * bins))


def _style_corner_ticks(ax, x_only=False):
    """Few, evenly spaced ticks that cannot collide across adjacent panels."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if not x_only:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
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
    """Return a publication-style parameter label for plot axes and titles."""
    if name.startswith("a") and name[1:].isdigit():
        return rf"$a_{{{name[1:]}}}$"
    labels = {
        "M_A [GeV]": r"$M_{A}\,[\mathrm{GeV}]$",
        "AxFFCCQEshape pull": r"$\mathrm{AxFFCCQEshape\ pull}$",
        "NormCCMEC pull": r"$\mathrm{NormCCMEC\ pull}$",
        "RPA CCQE pull": r"$\mathrm{RPA\ CCQE\ pull}$",
    }
    if name in labels:
        return labels[name]
    return name


# All corner PDFs are placed at the same 0.5\linewidth width in the paper.
# The two-parameter corner is the typography reference; larger source figures
# are shrunk more by LaTeX, so their source fonts must grow by the same factor.
CORNER_REFERENCE_WIDTH = 4.0


def _scale_figure_fonts(figure, scale):
    """Scale every existing text artist, including ticks and offset text."""
    for artist in figure.findobj(match=Text):
        artist.set_fontsize(artist.get_fontsize() * scale)


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
                              legend_headroom=.30,
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
    corner the contours already leave free.
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
    return fig


@mpl.rc_context(PUBLICATION_RC)
def plot_ma_posterior_overlay(results,
                              fit_keys=("ma_no_axff", "ma_uniform"),
                              labels=None, bins=45, figsize=(7.2, 7.2)):
    """Compare M_A, NormCCMEC, and RPA posteriors from compatible fits.

    Legend entries follow :func:`posterior_label` and name the prior whenever
    the selected fits use different priors; ``labels`` maps fit keys to
    replacement text.
    """
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    overlays = []
    expected_names = None
    for index, key in enumerate(fit_keys):
        if key not in results:
            raise KeyError(
                f"Posterior {key!r} is unavailable; run that XML and the "
                "notebook's main results cell first"
            )
        result = results[key]
        if "ma_joint_samples" not in result:
            raise ValueError(f"{key!r} is not a dipole-M_A fit")
        names = tuple(result["ma_joint_names"])
        samples = np.asarray(result["ma_joint_samples"])
        if names != ("M_A [GeV]", "NormCCMEC pull", "RPA CCQE pull"):
            raise ValueError(
                f"{key!r} has parameters {names}; expected M_A, NormCCMEC, "
                "and RPA_CCQE in that order"
            )
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("Selected M_A posteriors use different parameters")
        overlays.append((key, result["spec"], samples,
                         colors[index % len(colors)]))

    # Standard legend text; ``labels`` (keyed by fit key) overrides entries.
    labels = dict(labels or {})
    single_prior = len({prior_label(spec) for _, spec, _, _ in overlays}) == 1
    overlays = [
        (labels.get(key, posterior_label(spec, single_prior)), samples, color)
        for key, spec, samples, color in overlays
    ]

    n = len(expected_names)
    joined = np.concatenate([samples for _, samples, _ in overlays], axis=0)
    ranges = []
    for coordinate in range(n):
        low, high = np.quantile(joined[:, coordinate], [0.001, 0.999])
        padding = .04 * (high - low) if high > low else .5
        ranges.append((low - padding, high + padding))

    with mpl.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row < col:
                ax.set_visible(False)
                continue
            for label, samples, color in overlays:
                if row == col:
                    centers, density = _smooth_density(samples[:, col], bins)
                    ax.plot(centers, density, color=color, lw=2.0)
                else:
                    x_range, x_bins = _panel_binning(
                        ranges[col], samples[:, col], bins)
                    y_range, y_bins = _panel_binning(
                        ranges[row], samples[:, row], bins)
                    histogram, xedges, yedges = np.histogram2d(
                        samples[:, col], samples[:, row],
                        bins=[x_bins, y_bins], range=[x_range, y_range],
                    )
                    smoothed = _smooth_density_2d(histogram)
                    levels = _credible_density_levels(smoothed)
                    if levels:
                        xgrid, ygrid, density = _density_grid_to_edges(
                            xedges, yedges, smoothed
                        )
                        ax.contour(
                            xgrid, ygrid, density.T, levels=levels,
                            colors=color,
                            linewidths=np.linspace(1.2, 2.0, len(levels)),
                        )
            ax.set_xlim(ranges[col])
            if row > col:
                ax.set_ylim(ranges[row])
            if row == col:
                ax.set_yticks([])
            if row == n - 1:
                ax.set_xlabel(_parameter_label(expected_names[col]), fontsize=10.5)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(_parameter_label(expected_names[row]), fontsize=10.5)
            elif col > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=9)
            _style_corner_ticks(ax, x_only=row == col)
            ax.grid(color="#9AA4B2", alpha=.15, linewidth=.6)

    handles = [Line2D([], [], color=color, lw=1.8, label=label)
               for label, _, color in overlays]
    # Lay out the corner panels without the legend. An axes-attached legend
    # outside the first diagonal panel makes tight_layout reserve a large gap
    # between every column, dramatically shrinking the plots.
    fig.tight_layout(pad=.65)
    top_left = axes[0, 0].get_position()
    legend = fig.legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(top_left.x1 + .02, top_left.y1),
        fontsize=10.5, borderaxespad=0, handlelength=2.2,
    )
    legend.set_in_layout(False)
    return fig


@mpl.rc_context(PUBLICATION_FONT_RC)
def plot_corner(result, bins=35, show_all_coefficients=False,
                show_prior=False, prior_mask=None, axis_names=None,
                axis_ranges=None):
    """Joint physical posterior, restricted to independent coefficients by default.

    ``axis_ranges`` optionally maps a parameter name (as in ``result["names"]``,
    e.g. ``"M_A [GeV]"`` or ``"a1"``) to a fixed ``(low, high)`` display range,
    so panels for that parameter share one x/y range across fits and suites.
    Parameters absent from the mapping keep their automatic, sample-derived
    range. Pass ``None`` (the default) to fix no parameter's range.
    """
    # Nested blue regions follow the visual convention of the reference
    # corner plot: darker for 68%, lighter for 95%, without outline contours.
    posterior_color = "#0072C1"
    contour68_fill = "#7FB8DF"
    contour95_fill = "#BFDBEF"
    best_fit_color = "#000000"
    prior_color = "#D62728"
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
    figure_width = max(CORNER_REFERENCE_WIDTH, 1.75 * n)
    font_scale = figure_width / CORNER_REFERENCE_WIDTH
    fig, axes = plt.subplots(n, n, figsize=(figure_width, figure_width),
                             squeeze=False)
    ranges = []
    for i in range(n):
        if axis_ranges is not None and names[i] in axis_ranges:
            ranges.append(tuple(axis_ranges[names[i]]))
            continue
        range_samples = (np.r_[samples[:, i], prior_samples[:, i]]
                         if prior_mask[i] else samples[:, i])
        low, high = range_samples.min(), range_samples.max()
        padding = .03 * (high - low) if high > low else .5
        ranges.append((low - padding, high + padding))

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row < col:
                ax.set_visible(False)
                continue
            if row == col:
                centers, density = _smooth_density(samples[:, col], bins)
                ax.plot(centers, density, color=posterior_color, lw=1.8)
                if prior_mask[col]:
                    prior_centers, prior_density = _smooth_density(
                        prior_samples[:, col], bins
                    )
                    ax.plot(prior_centers, prior_density, color=prior_color,
                            lw=1.2, alpha=.6)
                ax.axvline(profile[col], color=best_fit_color, ls="--", lw=1.5)
                q16, median, q84 = _quantile(
                    samples[:, col], [.16, .50, .84]
                )
                central, minus, plus = _format_interval(
                    median, median - q16, q84 - median
                )
                if names[col].startswith("a"):
                    parameter = rf"a_{{{names[col][1:]}}}"
                elif names[col] == "M_A [GeV]":
                    parameter = r"M_{A}\,[\mathrm{GeV}]"
                elif names[col].endswith(" pull"):
                    base = names[col].removesuffix(" pull").replace(" ", r"\ ")
                    parameter = rf"\mathrm{{{base}}}\ \mathrm{{pull}}"
                else:
                    parameter = names[col]
                interval = (rf"{central}\pm{plus}" if minus == plus else
                            rf"{central}_{{-{minus}}}^{{+{plus}}}")
                ax.set_title(
                    rf"${parameter}={interval}$",
                    fontsize=9, pad=6,
                )
                ax.set_yticks([])
            else:
                x_range, x_bins = _panel_binning(
                    ranges[col], samples[:, col], bins)
                y_range, y_bins = _panel_binning(
                    ranges[row], samples[:, row], bins)
                histogram, xedges, yedges = np.histogram2d(
                    samples[:, col], samples[:, row], bins=[x_bins, y_bins],
                    range=[x_range, y_range],
                )
                histogram = _smooth_density_2d(histogram)
                levels = _credible_density_levels(histogram)
                if levels:
                    xgrid, ygrid, density = _density_grid_to_edges(
                        xedges, yedges, histogram
                    )
                    upper = np.nextafter(density.max(), np.inf)
                    if len(levels) == 2:
                        ax.contourf(
                            xgrid, ygrid, density.T,
                            levels=[levels[0], levels[1], upper],
                            colors=[contour95_fill, contour68_fill],
                        )
                    else:
                        ax.contourf(
                            xgrid, ygrid, density.T,
                            levels=[levels[0], upper], colors=[contour68_fill],
                        )
                if prior_mask[col] and prior_mask[row]:
                    prior_x_range, prior_x_bins = _panel_binning(
                        ranges[col], prior_samples[:, col], bins)
                    prior_y_range, prior_y_bins = _panel_binning(
                        ranges[row], prior_samples[:, row], bins)
                    prior_histogram, prior_xedges, prior_yedges = np.histogram2d(
                        prior_samples[:, col], prior_samples[:, row],
                        bins=[prior_x_bins, prior_y_bins],
                        range=[prior_x_range, prior_y_range],
                    )
                    prior_histogram = _smooth_density_2d(prior_histogram)
                    prior_levels = _credible_density_levels(prior_histogram)
                    if prior_levels:
                        prior_xgrid, prior_ygrid, prior_density = (
                            _density_grid_to_edges(
                                prior_xedges, prior_yedges, prior_histogram
                            )
                        )
                        ax.contour(
                            prior_xgrid, prior_ygrid, prior_density.T,
                            levels=prior_levels, colors=prior_color,
                            linewidths=np.linspace(.9, 1.3, len(prior_levels)),
                            linestyles=["--", "-"][-len(prior_levels):], alpha=.6,
                        )
                ax.plot(profile[col], profile[row],
                        "D", color=best_fit_color, markeredgecolor="white",
                        markeredgewidth=.5, markersize=5, zorder=4)
                ax.set_ylim(ranges[row])

            ax.set_xlim(ranges[col])
            _style_corner_ticks(ax, x_only=row == col)

            if row == n - 1:
                ax.set_xlabel(_parameter_label(axis_names[col]))
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(_parameter_label(axis_names[row]))
            elif col > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    legend_handles = [
        Patch(facecolor=contour68_fill, label="Posterior 68%"),
        Patch(facecolor=contour95_fill, label="Posterior 95%"),
        Line2D([], [], color=best_fit_color, marker="D", ls="none", markersize=6,
               label="Profile best fit"),
    ]
    if any_prior:
        legend_handles.extend([
            Line2D([], [], color=prior_color, lw=1.3, ls="-", alpha=.6,
                   label="Prior 68%"),
            Line2D([], [], color=prior_color, lw=1.0, ls="--", alpha=.6,
                   label="Prior 95%"),
        ])
    # Lay out the corner grid first, then put the legend in the otherwise unused
    # upper triangle. Keeping it out of the layout prevents it from shrinking the
    # diagonal panel beside it.
    # Scale after all axes text exists but before layout, so tight_layout uses
    # the compensated sizes. At 0.5\linewidth these render identically to the
    # fonts in the two-parameter (4-inch) reference corner.
    _scale_figure_fonts(fig, font_scale)
    fig.tight_layout(pad=.35)
    top_diagonal = axes[0, 0].get_position()
    # Four-parameter corners have enough unused upper-triangle space to move
    # the legend one panel farther right, clear of the second diagonal title.
    legend_x = (axes[0, 2].get_position().x0 + .015 if n == 4
                else top_diagonal.x1 + .012)
    legend = fig.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(legend_x, top_diagonal.y1),
        ncol=1, fontsize=7.5 * font_scale, frameon=False,
        labelspacing=.45, handlelength=1.8, handletextpad=.55,
        borderaxespad=0,
    )
    legend.set_in_layout(False)
    return fig


@mpl.rc_context(PUBLICATION_FONT_RC)
def plot_ma_nuisance_corner(result, bins=35, show_prior=False,
                            show_ma_prior=True, axis_ranges=None):
    """Joint posterior of physical M_A and its fitted cross-section nuisances."""
    joint = dict(result)
    n = len(result["ma_joint_names"])
    joint.update(
        samples=result["ma_joint_samples"],
        prior_samples=result["ma_joint_prior_samples"],
        profile=result["ma_joint_profile"],
        names=result["ma_joint_names"],
        joint_samples=result["ma_joint_samples"],
        joint_names=result["ma_joint_names"],
        joint_indices=np.arange(n),
    )
    return plot_corner(
        joint, bins=bins, show_prior=show_prior,
        prior_mask=[show_ma_prior] + [True] * (n - 1),
        axis_ranges=axis_ranges,
    )


def plot_zexp_nuisance_corner(result, bins=35, show_prior=False, axis_ranges=None):
    """Joint posterior of physical z coefficients and fitted nuisances."""
    joint = dict(result)
    n = len(result["zexp_nuisance_names"])
    joint.update(
        samples=result["zexp_nuisance_samples"],
        prior_samples=result["zexp_nuisance_prior_samples"],
        profile=result["zexp_nuisance_profile"],
        names=result["zexp_nuisance_names"],
        joint_samples=result["zexp_nuisance_samples"],
        joint_names=result["zexp_nuisance_names"],
        joint_indices=np.arange(n),
    )
    axis_names = [
        name.removesuffix(" pull") if name in {
            "NormCCMEC pull", "RPA CCQE pull"
        } else name
        for name in result["zexp_nuisance_names"]
    ]
    return plot_corner(
        joint, bins=bins, show_prior=show_prior, axis_names=axis_names,
        axis_ranges=axis_ranges,
    )


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
        if n_retained < 20_000:
            print("WARNING: fewer than 20,000 retained posterior samples; "
                  "also check effective sample size and convergence.")
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
    return results
