"""Split of PROfit's minimum chi2 into stat + syst (+ parameter pulls), for Neyman and CNP fits.

    chi2_tot = r^T C^-1 r  +  alpha^T alpha  +  theta^T theta
               (stat)         (syst)            (pull)

r is the residual of the drawn constrained best fit (``bestfit2d``), alpha the posterior pulls of
the covariance-type systematic modes (``covmode_pulls``), theta the Gaussian-constrained fitted
parameters. chi2_data = chi2_tot - pull comes from the log.

Neyman: C = diag(d), zero-data bins dropped. stat is computed from the histograms and
stat + syst reproduces chi2_data (to ~0.1), which is the closure check.

CNP: C_i = 3 / (1/d_i + 2/mu_i) (mu_i / 2 in a zero-data bin, which CNP keeps), with mu the best-fit
prediction BEFORE the covariance-mode shift. PROfile.root stores only the shifted one, so the exact
stat term is not available: stat is defined as chi2_data - syst, and the closure check becomes a
coarse guard, the same sum with mu reconstructed as

    mu_i = bestfit2d_i / (1 + covmode_bin_pulls_i * f_i),

f_i the fractional half-width of the pre-fit band in that bin (``preerrband_slice_ybinN``). On the
2026-11-24 production fits the guard closes within 1.0 (0.0 for fits that stay at the CV), against
up to 8 for the Neyman formula, hence the looser tolerance.
"""
import os
from pathlib import Path

import numpy as np
import uproot

CHANNEL = "nu_uBooNE_numuCC1p"
TOLERANCE = {"neyman": 0.5, "CNP": 1.5}


def fit_metric(path):
    """Statistic of the fits under ``path``: AXIAL_MASS_METRIC if set, otherwise CNP for an output
    tree with 'cnp' in its name and Neyman for the rest (same rule as notebook 26)."""
    metric = os.environ.get("AXIAL_MASS_METRIC") or ("CNP" if "cnp" in str(path).lower() else "neyman")
    if metric.lower() not in ("neyman", "cnp"):
        raise ValueError(f"chi2 split not implemented for metric {metric!r}")
    return "CNP" if metric.lower() == "cnp" else "neyman"


def stat_chi2(data, prediction, metric, mu=None):
    """Statistics-only chi2 of ``prediction`` against ``data``. For CNP the variance uses ``mu``
    (default: the prediction itself, which is exact when no covariance-mode shift is applied)."""
    data = np.asarray(data, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    residual2 = (data - prediction) ** 2
    use = data > 0
    if metric == "neyman":
        return float(np.sum(residual2[use] / data[use]))
    mu = prediction if mu is None else np.asarray(mu, dtype=float).reshape(-1)
    mu = np.maximum(mu, 1e-6)
    return float(np.sum(residual2[use] * (1 / data[use] + 2 / mu[use]) / 3) + np.sum(residual2[~use] / (mu[~use] / 2)))


def _unshifted_prediction(profile_file, bestfit, channel):
    """Best-fit prediction before the covariance-mode shift, reconstructed (see module docstring).
    None when the bin pulls or the pre-fit band slices are not in the file."""
    if "covmode_bin_pulls" not in profile_file:
        return None
    fractional = np.zeros(bestfit.shape)
    for y_bin in range(bestfit.shape[1]):
        key = f"{channel}_preerrband_slice_ybin{y_bin + 1}"
        if key not in profile_file:
            return None
        band = profile_file[key]
        centre = band.values(axis="y")
        half_width = 0.5 * (band.errors("low", axis="y") + band.errors("high", axis="y"))
        fractional[:, y_bin] = np.divide(half_width, centre, out=np.zeros_like(half_width), where=centre > 0)
    bin_pulls = profile_file["covmode_bin_pulls"]
    pulls = np.zeros(bestfit.size)
    pulls[np.floor(bin_pulls.values(axis="x")).astype(int)] = bin_pulls.values(axis="y")
    return bestfit.reshape(-1) / (1 + pulls * fractional.reshape(-1))


def stat_syst_split(profile_path, chi2_data, metric=None, channel=CHANNEL):
    """dict(stat, syst, mismatch, metric) for one fit; NaN where the inputs are missing.

    ``chi2_data`` is the log's total minus the parameter pull penalty. ``mismatch`` is the closure
    check: Neyman, stat + syst - chi2_data with stat from the histograms; CNP, the same with the
    reconstructed-mu stat, while the returned stat is chi2_data - syst. Compare to TOLERANCE[metric].
    """
    profile_path = Path(profile_path)
    metric = metric or fit_metric(profile_path)
    result = dict(stat=np.nan, syst=np.nan, mismatch=np.nan, metric=metric)
    if not profile_path.is_file():
        return result
    with uproot.open(profile_path) as profile_file:
        if "covmode_pulls" not in profile_file:
            return result
        data = profile_file[f"{channel}_data2d"].values().astype(float)
        bestfit = profile_file[f"{channel}_bestfit2d"].values().astype(float)
        syst = float(np.sum(profile_file["covmode_pulls"].values(axis="y") ** 2))
        mu = _unshifted_prediction(profile_file, bestfit, channel) if metric == "CNP" else None
    result["syst"] = syst
    if metric == "neyman":
        result["stat"] = stat_chi2(data, bestfit, metric)
        result["mismatch"] = result["stat"] + syst - chi2_data
    else:
        result["stat"] = chi2_data - syst
        if mu is not None:
            result["mismatch"] = stat_chi2(data, bestfit, metric, mu=mu) - result["stat"]
    return result


def warn_if_mismatch(result, label=""):
    """Print a warning when the closure check of ``stat_syst_split`` fails; returns True if it did."""
    tolerance = TOLERANCE[result["metric"]]
    if np.isfinite(result["mismatch"]) and abs(result["mismatch"]) > tolerance:
        print(f"  warning: {label} chi2 split closure off by {result['mismatch']:.2f} "
              f"(tolerance {tolerance}, {result['metric']})")
        return True
    return False


def stat_variance(data_variance, prediction, metric):
    """Per-bin statistical variance of the fit statistic for a chi2 against ``prediction``.
    Neyman: the data variance itself. CNP: 3 / (1/d + 2/mu), mu / 2 where the data variance is 0;
    exact for a prediction without covariance-mode shift (the pre-fit CV)."""
    data_variance = np.asarray(data_variance, dtype=float)
    if metric == "neyman":
        return data_variance
    mu = np.maximum(np.asarray(prediction, dtype=float), 1e-6)
    safe = np.where(data_variance > 0, data_variance, 1.0)
    return np.where(data_variance > 0, 3.0 / (1.0 / safe + 2.0 / mu), mu / 2.0)
