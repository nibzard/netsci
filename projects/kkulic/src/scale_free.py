"""Scale-free / degree-distribution analysis.

Fits a power law to the empirical degree distribution using Clauset-Shalizi-
Newman maximum-likelihood estimation (via the ``powerlaw`` package),
compares it against alternative distributions (exponential, log-normal),
and reports a Kolmogorov-Smirnov goodness-of-fit statistic. Because the
network here is built by thresholding a similarity graph rather than by a
generative preferential-attachment process, a true power law is not
guaranteed -- the analysis is designed to report that result honestly
rather than assume it.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import powerlaw

logger = logging.getLogger(__name__)


def degree_sequence(g: nx.Graph) -> np.ndarray:
    degrees = np.array([d for _, d in g.degree()])
    return degrees[degrees > 0]


def fit_power_law(degrees: np.ndarray) -> powerlaw.Fit:
    return powerlaw.Fit(degrees, discrete=True, verbose=False)


def compare_distributions(fit: powerlaw.Fit) -> dict:
    """Compare power law against exponential and log-normal alternatives.

    Returns a dict of {comparison_name: (loglikelihood_ratio, p_value)}.
    Positive R favors power law; negative favors the alternative. Comparisons
    that fail to converge (e.g. too few unique degree values, as can happen
    on small or near-homogeneous-degree graphs) are reported as NaN rather
    than raising, since that itself is informative about scale-freeness.
    """
    results = {}
    for alt in ("exponential", "lognormal", "stretched_exponential"):
        try:
            R, p = fit.distribution_compare("power_law", alt)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Distribution comparison power_law vs %s failed: %s", alt, exc)
            R, p = float("nan"), float("nan")
        results[alt] = (R, p)
    return results


def summarize(g: nx.Graph) -> dict:
    degrees = degree_sequence(g)
    n_unique = len(np.unique(degrees))
    if len(degrees) < 10 or n_unique < 2:
        logger.warning(
            "Degree sequence too small or too homogeneous (%d nodes, %d unique "
            "degrees) for a reliable power-law fit; results will likely be NaN. "
            "This is expected on small/near-complete graphs and is itself "
            "evidence against scale-free structure.",
            len(degrees), n_unique,
        )
    fit = fit_power_law(degrees)
    comparisons = compare_distributions(fit)
    try:
        alpha, xmin, ks = fit.power_law.alpha, fit.power_law.xmin, fit.power_law.D
    except (ValueError, RuntimeError) as exc:
        logger.warning("Power-law parameter fit failed: %s", exc)
        alpha, xmin, ks = float("nan"), float("nan"), float("nan")
    return {
        "alpha": alpha,
        "xmin": xmin,
        "ks_distance": ks,
        "comparisons": comparisons,
        "n_nodes": g.number_of_nodes(),
        "mean_degree": float(degrees.mean()) if len(degrees) else float("nan"),
    }
