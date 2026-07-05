"""Centrality analysis for the Croatian tourism similarity network."""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def _eigenvector_centrality_per_component(g: nx.Graph) -> dict:
    """``eigenvector_centrality_numpy`` refuses disconnected graphs (the
    leading eigenvector is only defined up to an arbitrary per-component
    scale). Run it independently on each connected component instead,
    which is well-defined and is the standard workaround.
    """
    scores: dict = {}
    for component in nx.connected_components(g):
        sub = g.subgraph(component)
        if len(sub) == 1:
            scores[next(iter(sub.nodes()))] = 0.0
            continue
        scores.update(nx.eigenvector_centrality_numpy(sub, weight="weight"))
    return scores


def compute_centralities(g: nx.Graph) -> pd.DataFrame:
    """Compute degree, weighted strength, betweenness, closeness, eigenvector
    centrality and PageRank for every node; return a sorted DataFrame.
    """
    degree = dict(g.degree())
    strength = dict(g.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(g, weight="weight", normalized=True)
    closeness = nx.closeness_centrality(g, distance=None)
    try:
        eigenvector = nx.eigenvector_centrality(g, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality failed to converge; falling back to numpy variant.")
        eigenvector = _eigenvector_centrality_per_component(g)
    pagerank = nx.pagerank(g, weight="weight")

    df = pd.DataFrame(
        {
            "region": list(g.nodes()),
            "degree": [degree[n] for n in g.nodes()],
            "strength": [strength[n] for n in g.nodes()],
            "betweenness": [betweenness[n] for n in g.nodes()],
            "closeness": [closeness[n] for n in g.nodes()],
            "eigenvector": [eigenvector[n] for n in g.nodes()],
            "pagerank": [pagerank[n] for n in g.nodes()],
        }
    )
    return df.sort_values("pagerank", ascending=False).reset_index(drop=True)


def top_hubs(df: pd.DataFrame, metric: str = "pagerank", n: int = 10) -> pd.DataFrame:
    return df.sort_values(metric, ascending=False).head(n)


def centrality_correlations(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["degree", "strength", "betweenness", "closeness", "eigenvector", "pagerank"]
    return df[metrics].corr(method="spearman")
