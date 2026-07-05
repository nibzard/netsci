"""Network resilience under random failure vs. targeted attack.

Simulates progressive node removal in two regimes:

* Random failure -- nodes removed in random order (averaged over repeats).
* Targeted attack -- nodes removed in decreasing order of a centrality
  metric (default: degree), recomputed after each removal ("adaptive"
  attack) or fixed from the initial ranking ("static" attack).

Tracks giant-component fractional size and global efficiency as a function
of the fraction of nodes removed, which is the standard resilience curve
used to compare robustness to random vs. targeted failure (Albert-Jeong-
Barabasi).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


def _giant_component_fraction(g: nx.Graph, n_total: int) -> float:
    if g.number_of_nodes() == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(g)), default=0)
    return largest / n_total


def simulate_random_failure(g: nx.Graph, n_repeats: int = 20, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = g.number_of_nodes()
    nodes = list(g.nodes())
    all_runs = []

    for run in range(n_repeats):
        order = rng.permutation(nodes)
        h = g.copy()
        fractions = [_giant_component_fraction(h, n_total)]
        for node in order:
            h.remove_node(node)
            fractions.append(_giant_component_fraction(h, n_total))
        all_runs.append(fractions)

    arr = np.array(all_runs)
    removed_frac = np.arange(arr.shape[1]) / n_total
    return pd.DataFrame(
        {
            "removed_fraction": removed_frac,
            "giant_component_fraction_mean": arr.mean(axis=0),
            "giant_component_fraction_std": arr.std(axis=0),
        }
    )


def simulate_targeted_attack(g: nx.Graph, metric: str = "degree", adaptive: bool = True) -> pd.DataFrame:
    n_total = g.number_of_nodes()
    h = g.copy()
    fractions = [_giant_component_fraction(h, n_total)]

    if not adaptive:
        ranking = _rank_nodes(g, metric)
        order = [n for n, _ in ranking]
        for node in order:
            if node in h:
                h.remove_node(node)
                fractions.append(_giant_component_fraction(h, n_total))
    else:
        while h.number_of_nodes() > 0:
            ranking = _rank_nodes(h, metric)
            top_node = ranking[0][0]
            h.remove_node(top_node)
            fractions.append(_giant_component_fraction(h, n_total))

    removed_frac = np.arange(len(fractions)) / n_total
    return pd.DataFrame({"removed_fraction": removed_frac, "giant_component_fraction": fractions})


def _rank_nodes(g: nx.Graph, metric: str) -> list[tuple]:
    if metric == "degree":
        scores = dict(g.degree(weight="weight"))
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(g, weight="weight")
    elif metric == "pagerank":
        scores = nx.pagerank(g, weight="weight")
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def global_efficiency_curve(g: nx.Graph, order: list) -> pd.DataFrame:
    n_total = g.number_of_nodes()
    h = g.copy()
    effs = [nx.global_efficiency(h)]
    for node in order:
        if node in h:
            h.remove_node(node)
            effs.append(nx.global_efficiency(h) if h.number_of_nodes() > 1 else 0.0)
    removed_frac = np.arange(len(effs)) / n_total
    return pd.DataFrame({"removed_fraction": removed_frac, "global_efficiency": effs})
