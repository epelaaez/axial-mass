"""Importance reweighting of PROfit MCMC chains to the exact z-expansion response.

PROfit combines several spline nuisance parameters *multiplicatively*: the
predicted content of every bin is the central value times a product of
independent one-dimensional splines, one per parameter
(``PROsyst::GetSplineShiftedSpectrum``).  For the z-expansion parameters
``eta = (eta_1, ..., eta_n)`` (standardized PCA coordinates of the prior
covariance) the true per-bin response is an exact multivariate quadratic with
cross terms ``eta_i eta_j``; the product of one-dimensional splines replaces the
true cross term by a spurious ``b_i b_j`` term.  The resulting error in the data
term of the chi-squared,

    dchi2_data(eta) = chi2_data(mu_spline(eta)) - chi2_data(mu_exact(eta)),

is tabulated on grids in eta space by
``python/notebooks/12_spline_factorization_validation.ipynb`` and written to
``tables/spline_factorization/grids/{fit}_{suite}_{grid_type}_dchi2.npz``.

Because the Gaussian pull term is identical in the spline and exact models, the
ratio of the exact to the sampled posterior density at a chain sample is

    w(eta) = exp(+dchi2_data(eta) / 2),

so every stored chain can be corrected by importance reweighting without
re-running any fit.  This module provides the grid loader, the weight
computation, the affine map from eta to z-expansion coefficients, weighted
summary statistics, and a chain loader.  :mod:`postfit_physical_parameters`
applies the weights to every fit it loads, so all downstream posterior
quantities (means, widths, credible intervals, corner plots, F_A bands, derived
quantities) are corrected consistently.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


REPO_MA_ZEXP = Path(__file__).resolve().parents[2]
GRID_DIR = REPO_MA_ZEXP / "tables" / "spline_factorization" / "grids"
GRID_PRODUCER = "python/notebooks/12_spline_factorization_validation.ipynb"

# Suite names as used by the notebooks (``nuwro_fit_results``), by notebook 12's
# tables (``nuwro``), by the on-disk productions (``zexp_prior_fits``) and by the
# request that motivated this module (``open_data``).  All map to the short
# name used in the grid file names.
SUITE_ALIASES = {
    "nuwro": "nuwro",
    "nuwro_fit_results": "nuwro",
    "zexp_prior_fits": "nuwro",
    "asimov": "asimov",
    "asimov_fit_results": "asimov",
    "zexp_prior_fits_asimov": "asimov",
    "opendata": "opendata",
    "open_data": "opendata",
    "opendata_fit_results": "opendata",
    "zexp_prior_fits_opendata": "opendata",
}
SUITE_FIT_RESULTS = {
    "nuwro": "nuwro_fit_results",
    "asimov": "asimov_fit_results",
    "opendata": "opendata_fit_results",
}

# Diagnostics (warnings, never errors).
ESS_WARNING_FRACTION = 0.5       # warn when ESS < 0.5 * N
GAUSSIAN_DCHI2_WARNING = 2.0     # warn when |dchi2| > 2 inside the 95% chain core of a Gaussian-prior fit
CHAIN_CORE_FRACTION = 0.95


class SplineReweightingWarning(UserWarning):
    """Diagnostic emitted by the spline-factorization reweighting."""


def normalize_suite(suite):
    """Return the short suite name (``nuwro``, ``asimov`` or ``opendata``)."""
    key = str(suite).strip().lower()
    if key not in SUITE_ALIASES:
        raise KeyError(
            f"Unknown suite {suite!r}; choose from {sorted(set(SUITE_ALIASES))}"
        )
    return SUITE_ALIASES[key]


def default_grid_type(fit_name):
    """``extended`` for uniform-prior fits (their chains roam outside [-3, 3]), else ``full``."""
    return "extended" if "uniform" in fit_name else "full"


def grid_path(fit_name, suite, grid_type="auto"):
    """Path of the saved grid for one fit, suite and grid type."""
    if grid_type == "auto":
        grid_type = default_grid_type(fit_name)
    return GRID_DIR / f"{fit_name}_{normalize_suite(suite)}_{grid_type}_dchi2.npz"


def load_dchi2_grid(fit_name, suite, grid_type="auto"):
    """Load the tabulated ``dchi2_data(eta)`` grid for one fit and suite.

    Parameters
    ----------
    fit_name : str
        Fit key, e.g. ``'minerva_k6'``, ``'lqcd_k6'``, ``'minerva_k6_uniform'``.
    suite : str
        ``'nuwro'``, ``'asimov'``, ``'open_data'`` (or any alias in
        :data:`SUITE_ALIASES`, such as ``'nuwro_fit_results'``).
    grid_type : {'auto', 'full', 'extended'}
        ``'auto'`` picks ``'extended'`` ([-10, 10]^n) when ``'uniform'`` is in
        the fit name and ``'full'`` ([-3, 3]^n) otherwise.

    Returns
    -------
    dict
        ``eta_axes`` (list of 1D coordinate arrays, one per PCA parameter),
        ``dchi2`` (N-D array on the meshgrid of those axes, ``ij`` indexing),
        ``fit_name``, ``suite`` (short name), ``grid_type``, plus every other
        scalar or array stored in the file (``eta_names``, ``prior``,
        ``dchi2_b2``, ``chi2_exact``, ``max_frac``, ``has_chain``, ...).
    """
    if grid_type == "auto":
        grid_type = default_grid_type(fit_name)
    if grid_type not in ("full", "extended"):
        raise ValueError(f"grid_type must be 'auto', 'full' or 'extended', not {grid_type!r}")
    suite = normalize_suite(suite)
    path = grid_path(fit_name, suite, grid_type)
    if not path.is_file():
        available = sorted(p.name for p in GRID_DIR.glob("*_dchi2.npz")) if GRID_DIR.is_dir() else []
        raise FileNotFoundError(
            f"No saved dchi2 grid for fit={fit_name!r}, suite={suite!r}, grid_type={grid_type!r}: "
            f"{path} does not exist. Run the grid-saving cells at the end of {GRID_PRODUCER} first. "
            f"Grids currently available: {available or 'none'}"
        )
    with np.load(path, allow_pickle=False) as stored:
        grid = {key: stored[key] for key in stored.files}
    eta_axes = [np.asarray(axis, dtype=float) for axis in grid.pop("eta_axes")]
    dchi2 = np.asarray(grid.pop("dchi2"), dtype=float)
    if dchi2.shape != tuple(len(axis) for axis in eta_axes):
        raise ValueError(
            f"{path}: dchi2 shape {dchi2.shape} does not match the axes "
            f"{tuple(len(axis) for axis in eta_axes)}"
        )
    # Unwrap 0-d arrays (strings, scalars) into Python objects.
    for key, value in list(grid.items()):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            grid[key] = value.item()
    grid.update(eta_axes=eta_axes, dchi2=dchi2, fit_name=fit_name, suite=suite,
                grid_type=grid_type, path=path)
    return grid


def _is_gaussian_prior(grid_dict, gaussian_prior=None):
    if gaussian_prior is not None:
        return bool(gaussian_prior)
    return "uniform" not in str(grid_dict.get("fit_name", ""))


def chain_core_mask(eta_samples, fraction=CHAIN_CORE_FRACTION):
    """Boolean mask of the ``fraction`` of samples closest to the mean (Mahalanobis distance).

    Used as a cheap stand-in for "inside the 95% credible region" when checking
    how large the factorization error is where the posterior actually lives.
    """
    eta = np.atleast_2d(np.asarray(eta_samples, dtype=float))
    centered = eta - eta.mean(axis=0)
    covariance = np.atleast_2d(np.cov(eta, rowvar=False))
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(covariance)
    distance = np.einsum("ni,ij,nj->n", centered, inverse, centered)
    threshold = np.quantile(distance, fraction)
    return distance <= threshold


def effective_sample_size(weights):
    """Kish effective sample size ``(sum w)^2 / sum w^2``."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.sum(weights ** 2))


def _gpdfit(excess):
    """Empirical-Bayes fit of a generalized Pareto distribution (Zhang & Stephens 2009).

    ``excess`` are the tail exceedances, sorted ascending and positive.  Returns
    ``(k, sigma)``, the shape and scale parameters.  Transcribed from the
    reference implementation (``arviz.stats.stats._gpdfit``, i.e. ``psis.py`` of
    Vehtari, Simpson, Gelman, Yao & Gabry, arXiv:1507.02646), and checked
    against it to 1e-12 on the chains of this analysis.
    """
    prior_b, prior_k = 3, 10
    n = len(excess)
    n_grid = 30 + int(n ** 0.5)

    b_grid = 1 - np.sqrt(n_grid / (np.arange(1, n_grid + 1, dtype=float) - 0.5))
    b_grid /= prior_b * excess[int(n / 4 + 0.5) - 1]
    b_grid += 1 / excess[-1]

    k_grid = np.log1p(-b_grid[:, None] * excess).mean(axis=1)
    log_likelihood = n * (np.log(-(b_grid / k_grid)) - k_grid - 1)
    with np.errstate(over="ignore"):     # the profile likelihood underflows far from the mode
        weights = 1 / np.exp(log_likelihood - log_likelihood[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps
    weights, b_grid = weights[keep], b_grid[keep]
    weights /= weights.sum()

    b_post = np.sum(b_grid * weights)
    k_post = np.log1p(-b_post * excess).mean()
    sigma = -k_post / b_post
    return (n * k_post + prior_k * 0.5) / (n + prior_k), sigma


def pareto_khat(log_weights, reff=1.0):
    """Pareto shape parameter ``k`` of the importance weights (PSIS diagnostic).

    A generalized Pareto distribution is fitted to the tail of the weights, the
    tail being the largest ``min(N / 5, 3 sqrt(N / reff))`` of them as in
    ``arviz.psislw``.  The fitted shape parameter diagnoses the reliability of
    the importance sampling: ``k < 0.5`` means the weights have finite variance
    and the estimates converge at the usual rate, ``0.5 <= k < 0.7`` is still
    usable in practice, and ``k >= 0.7`` means the estimates should not be
    trusted (Vehtari et al., arXiv:1507.02646).  The smoothing of the tail
    weights that PSIS also performs is not applied here; only ``k`` is returned.
    """
    x = np.asarray(log_weights, dtype=float).copy()
    n = len(x)
    x -= x.max()                                     # for numerical accuracy, as in psislw
    cutoff_index = -int(np.ceil(min(n / 5.0, 3 * (n / reff) ** 0.5))) - 1
    cutoff = max(np.sort(x)[cutoff_index], np.log(np.finfo(float).tiny))
    tail = x[x > cutoff]
    if len(tail) <= 4:                               # too short a tail to fit
        return float("inf")
    excess = np.sort(np.exp(tail) - np.exp(cutoff))
    return float(_gpdfit(excess)[0])


def uniform_weights(n_samples):
    """Normalized uniform weights, for fits that need no correction (dipole M_A)."""
    return np.full(int(n_samples), 1.0 / int(n_samples))


def compute_importance_weights(eta_samples, grid_dict, gaussian_prior=None, verbose=True):
    """Importance weights that turn a spline-model chain into the exact-model posterior.

    Parameters
    ----------
    eta_samples : array, shape (N_samples, N_pca)
        Chain samples in eta space (the ``weight_spline_FAzexp...PCAi`` columns,
        in the order of the prior's ``variation_branches``).
    grid_dict : dict
        Output of :func:`load_dchi2_grid`.
    gaussian_prior : bool, optional
        Whether the fit has a Gaussian prior on eta (decides whether the
        ``|dchi2| > 2`` diagnostic applies).  Inferred from the fit name when
        omitted.
    verbose : bool
        Emit the diagnostic warnings.

    Returns
    -------
    weights_raw : array, shape (N_samples,)
        ``exp(+dchi2_data(eta_k) / 2)``.
    weights_norm : array, shape (N_samples,)
        ``weights_raw / weights_raw.sum()``.
    ess : float
        Kish effective sample size ``(sum w)^2 / sum w^2``.
    dchi2_samples : array, shape (N_samples,)
        ``dchi2_data`` interpolated onto every sample.

    Notes
    -----
    ``dchi2_data`` is interpolated with ``RegularGridInterpolator(method='linear',
    bounds_error=False, fill_value=None)``; ``fill_value=None`` extrapolates
    linearly outside the grid, which only matters for uniform-prior chains whose
    ``restrict`` range exceeds the saved grid (the extended grid covers [-10, 10]).
    """
    eta = np.asarray(eta_samples, dtype=float)
    if eta.ndim == 1:
        eta = eta[:, None]
    axes = grid_dict["eta_axes"]
    if eta.shape[1] != len(axes):
        raise ValueError(
            f"eta_samples has {eta.shape[1]} columns but the grid for "
            f"{grid_dict.get('fit_name')} has {len(axes)} PCA parameters"
        )
    interpolator = RegularGridInterpolator(
        axes, grid_dict["dchi2"], method="linear", bounds_error=False, fill_value=None,
    )
    dchi2_samples = np.asarray(interpolator(eta), dtype=float)
    if not np.all(np.isfinite(dchi2_samples)):
        raise ValueError("non-finite dchi2 values after interpolation; check the grid and the samples")

    log_weights = 0.5 * dchi2_samples
    if log_weights.max() > 700.0:
        # exp would overflow. The normalized weights and the ESS are invariant
        # under a common factor, so shift the exponent and say so.
        warnings.warn(
            f"{grid_dict.get('fit_name')}/{grid_dict.get('suite')}: max dchi2/2 = {log_weights.max():.1f} "
            "would overflow exp(); raw weights are rescaled by exp(-max(dchi2/2)).",
            SplineReweightingWarning, stacklevel=2,
        )
        log_weights = log_weights - log_weights.max()
    weights_raw = np.exp(log_weights)
    weights_norm = weights_raw / weights_raw.sum()
    ess = effective_sample_size(weights_raw)

    if verbose:
        label = f"{grid_dict.get('fit_name')}/{grid_dict.get('suite')} ({grid_dict.get('grid_type')} grid)"
        n = len(eta)
        if ess < ESS_WARNING_FRACTION * n:
            warnings.warn(
                f"{label}: importance reweighting is significantly degrading sample quality: "
                f"ESS = {ess:.0f} of {n} samples ({100 * ess / n:.1f}%).",
                SplineReweightingWarning, stacklevel=2,
            )
        if _is_gaussian_prior(grid_dict, gaussian_prior) and n > 1:
            core = chain_core_mask(eta)
            worst = float(np.max(np.abs(dchi2_samples[core])))
            if worst > GAUSSIAN_DCHI2_WARNING:
                warnings.warn(
                    f"{label}: the factorization error is larger than expected for a Gaussian-prior "
                    f"fit: max |dchi2_data| = {worst:.2f} inside the {100 * CHAIN_CORE_FRACTION:.0f}% chain "
                    f"region (expected < {GAUSSIAN_DCHI2_WARNING}).",
                    SplineReweightingWarning, stacklevel=2,
                )
        outside = np.any((eta < np.array([a[0] for a in axes])) | (eta > np.array([a[-1] for a in axes])), axis=1)
        if outside.any():
            warnings.warn(
                f"{label}: {100 * outside.mean():.2f}% of the samples lie outside the saved grid and use "
                "linear extrapolation of dchi2.",
                SplineReweightingWarning, stacklevel=2,
            )
    return weights_raw, weights_norm, ess, dchi2_samples


def eta_to_a_coefficients(eta_samples, a_cv, delta_a_matrix):
    """Map eta samples to complete z-expansion coefficient vectors.

    ``a(eta) = a_cv + eta @ delta_a_matrix`` with ``a_cv`` of shape
    ``(N_coeffs,)`` and ``delta_a_matrix`` of shape ``(N_pca, N_coeffs)`` whose
    row ``i`` is the completed coefficient shift for one unit step along PCA
    direction ``i`` (the dependent coefficients re-solved from the sum rules
    and ``F_A(0)``).  The map is exactly affine because the constraints are
    linear.  Returns an array of shape ``(N_samples, N_coeffs)``.
    """
    eta = np.asarray(eta_samples, dtype=float)
    if eta.ndim == 1:
        eta = eta[:, None]
    a_cv = np.asarray(a_cv, dtype=float)
    delta = np.asarray(delta_a_matrix, dtype=float)
    if delta.shape != (eta.shape[1], len(a_cv)):
        raise ValueError(
            f"delta_a_matrix must have shape (N_pca={eta.shape[1]}, N_coeffs={len(a_cv)}), not {delta.shape}"
        )
    return a_cv[None, :] + eta @ delta


def zexp_affine_map(prior):
    """``(a_cv, delta_a_matrix)`` for a :class:`ZExpPrior`, in the layout of :func:`eta_to_a_coefficients`.

    Uses the same PCA convention as the stored PROfit branches
    (:func:`postfit_physical_parameters._zexp_transform`).
    """
    from postfit_physical_parameters import _zexp_transform  # local import: that module imports this one

    central, matrix = _zexp_transform(prior)
    return np.asarray(central, dtype=float), np.asarray(matrix, dtype=float).T


def _normalized(weights, n_samples):
    if weights is None:
        return uniform_weights(n_samples)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_samples,):
        raise ValueError(f"weights must have shape ({n_samples},), not {weights.shape}")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must not all be zero")
    return weights / total


def weighted_quantile(samples, weights_norm, quantiles):
    """Weighted quantiles, column by column.

    Values are sorted, the cumulative weight is accumulated, and the quantile
    is read where the cumulative weight (mid-point convention, so uniform
    weights reproduce ``numpy.quantile`` closely) crosses each requested level.
    ``samples`` may be ``(N,)`` or ``(N, N_cols)``; the result has shape
    ``(len(quantiles),)`` or ``(len(quantiles), N_cols)``.
    """
    x = np.asarray(samples, dtype=float)
    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((q < 0) | (q > 1)):
        raise ValueError("quantiles must lie in [0, 1]")
    one_dimensional = x.ndim == 1
    if one_dimensional:
        x = x[:, None]
    w = _normalized(weights_norm, x.shape[0])
    out = np.empty((len(q), x.shape[1]))
    for column in range(x.shape[1]):
        order = np.argsort(x[:, column], kind="mergesort")
        sorted_values = x[order, column]
        sorted_weights = w[order]
        cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        out[:, column] = np.interp(q, cumulative, sorted_values)
    return out[:, 0] if one_dimensional else out


def weighted_mean_std(samples, weights_norm):
    """Weighted mean and (population) standard deviation, column by column."""
    x = np.asarray(samples, dtype=float)
    w = _normalized(weights_norm, x.shape[0])
    mean = np.average(x, weights=w, axis=0)
    std = np.sqrt(np.average((x - mean) ** 2, weights=w, axis=0))
    return mean, std


def weighted_covariance(samples, weights_norm):
    """Weighted covariance matrix (``numpy.cov`` with ``aweights``, unbiased normalization)."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    w = _normalized(weights_norm, x.shape[0])
    return np.atleast_2d(np.cov(x, rowvar=False, aweights=w))


def weighted_summary(samples, weights_norm, credible_levels=(0.68, 0.95), names=None):
    """Weighted mean, standard deviation and central credible intervals.

    Parameters
    ----------
    samples : array, shape (N,) or (N, N_cols)
    weights_norm : array, shape (N,)
        Normalized importance weights (``None`` for uniform weights).
    credible_levels : iterable of float
        Central credible levels, e.g. ``(0.68, 0.95)``.
    names : sequence of str, optional
        Row labels (one per column of ``samples``).

    Returns
    -------
    pandas.DataFrame
        One row per column of ``samples`` with ``mean``, ``std``, ``median`` and,
        for every level ``L`` (as a percentage), ``low_L``, ``high_L`` (the
        equal-tailed interval) and ``half_width_L`` (half its length).
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    levels = tuple(float(level) for level in credible_levels)
    mean, std = weighted_mean_std(x, weights_norm)
    quantiles = [0.5]
    for level in levels:
        tail = (1.0 - level) / 2.0
        quantiles.extend([tail, 1.0 - tail])
    values = weighted_quantile(x, weights_norm, quantiles)
    table = {"mean": mean, "std": std, "median": values[0]}
    for index, level in enumerate(levels):
        low, high = values[1 + 2 * index], values[2 + 2 * index]
        tag = f"{100 * level:.0f}"
        table[f"low_{tag}"] = low
        table[f"high_{tag}"] = high
        table[f"half_width_{tag}"] = (high - low) / 2.0
    if names is None:
        names = [f"x{i}" for i in range(x.shape[1])] if x.shape[1] > 1 else ["x"]
    frame = pd.DataFrame(table, index=list(names))
    frame.index.name = "parameter"
    return frame


def _fit_spec(fit_name):
    from postfit_physical_parameters import SPECS  # local import: that module imports this one

    for spec in SPECS:
        if spec.key == fit_name:
            return spec
    raise KeyError(f"Unknown fit {fit_name!r}; known fits: {[spec.key for spec in SPECS]}")


def locate_chain_file(fit_name, suite):
    """Path of the ``*_v1_PROfile.root`` file holding the MCMC chain, or ``None``.

    The production the other notebooks read (``SUITE_DATA_DIRS``) is searched
    first; fits absent there (``lqcd_k7``, ``minerva_lqcd_k7``) are looked up in
    the newer production that carries the suite's own name, as notebook 12 does.
    """
    from postfit_physical_parameters import DATA_ROOT, SUITE_DATA_DIRS  # local import

    suite_key = SUITE_FIT_RESULTS[normalize_suite(suite)]
    for production in (SUITE_DATA_DIRS.get(suite_key, suite_key), suite_key):
        directory = DATA_ROOT / production / fit_name
        preferred = directory / f"{fit_name}_v1_PROfile.root"
        if preferred.is_file():
            return preferred
        matches = sorted(directory.glob("*_v1_PROfile.root"))
        if len(matches) == 1:
            return matches[0]
    return None


def load_chain(fit_name, suite, burn_in=0, thin=1):
    """Load the MCMC chain of one z-expansion fit.

    Returns ``(chain_df, eta_samples)``: the full chain as a DataFrame (one
    column per stored branch, float64) and the eta columns
    (``spec.prior.variation_branches``, i.e. ``weight_spline_FAzexp...PCAi``)
    as an array of shape ``(N_samples, N_pca)``.  Returns ``(None, None)`` with
    a warning when no chain file exists for this fit and suite.
    """
    import uproot

    spec = _fit_spec(fit_name)
    if spec.prior is None:
        raise ValueError(f"{fit_name!r} is a dipole-M_A fit and has no eta parameters")
    path = locate_chain_file(fit_name, suite)
    if path is None:
        warnings.warn(
            f"No MCMC chain found for fit={fit_name!r}, suite={normalize_suite(suite)!r}.",
            SplineReweightingWarning, stacklevel=2,
        )
        return None, None
    with uproot.open(path) as root:
        chains = [key.split(";")[0] for key in root.keys() if key.split(";")[0].endswith("_mcmc_chain")]
        if len(chains) != 1:
            raise RuntimeError(f"Expected one MCMC chain in {path}, found {chains}")
        tree = root[chains[0]]
        chain_df = pd.DataFrame({branch: tree[branch].array(library="np").astype(float) for branch in tree.keys()})
    chain_df = chain_df.iloc[burn_in::thin].reset_index(drop=True)
    chain_df.attrs.update(fit_name=fit_name, suite=normalize_suite(suite), path=str(path))
    eta_columns = list(spec.prior.variation_branches)
    missing = [column for column in eta_columns if column not in chain_df.columns]
    if missing:
        raise KeyError(f"{path} lacks the eta branches {missing}; stored branches: {list(chain_df.columns)}")
    eta_samples = chain_df[eta_columns].to_numpy(dtype=float)
    return chain_df, eta_samples


def reweight_fit(fit_name, suite, burn_in=0, thin=1, grid_type="auto", verbose=True):
    """Convenience wrapper: chain, grid and weights in one call.

    Returns a dict with ``chain_df``, ``eta_samples``, ``grid``, ``weights_raw``,
    ``weights_norm``, ``ess``, ``dchi2_samples``, or ``None`` when no chain exists.
    """
    chain_df, eta_samples = load_chain(fit_name, suite, burn_in=burn_in, thin=thin)
    if chain_df is None:
        return None
    grid = load_dchi2_grid(fit_name, suite, grid_type=grid_type)
    weights_raw, weights_norm, ess, dchi2_samples = compute_importance_weights(eta_samples, grid, verbose=verbose)
    return dict(chain_df=chain_df, eta_samples=eta_samples, grid=grid, weights_raw=weights_raw,
                weights_norm=weights_norm, ess=ess, dchi2_samples=dchi2_samples)


def describe_ess(ess, n_samples):
    """One-line ESS report used by the notebooks."""
    return f"Reweighting ESS: {ess:.0f} / {n_samples} ({100 * ess / n_samples:.1f}%)"
