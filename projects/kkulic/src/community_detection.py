"""Community detection on the tourism similarity network.

Primary algorithm: Louvain modularity maximization (python-louvain). Also
provides Girvan-Newman (for small/medium graphs, as a comparison method
covered in the course) and modularity-based evaluation against the
coastal/continental ground-truth label via Normalized Mutual Information.
"""
from __future__ import annotations

import itertools
import logging

import community as community_louvain
import networkx as nx
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

logger = logging.getLogger(__name__)


def louvain_partition(g: nx.Graph, resolution: float = 1.0, random_state: int = 42) -> dict:
    return community_louvain.best_partition(g, weight="weight", resolution=resolution, random_state=random_state)


def girvan_newman_partition(g: nx.Graph, n_communities: int = 4) -> dict:
    """Run Girvan-Newman until ``n_communities`` communities are found.

    Quadratic-ish cost via edge betweenness recomputation; intended for
    comparison on the pooled graph, not for large temporal graphs.
    """
    comp_gen = nx.algorithms.community.girvan_newman(g)
    for communities in itertools.islice(comp_gen, n_communities - 1):
        partition_sets = communities
    labels = {}
    for idx, comm in enumerate(partition_sets):
        for node in comm:
            labels[node] = idx
    return labels


def partition_to_frame(g: nx.Graph, partition: dict, label_col: str = "community") -> pd.DataFrame:
    rows = []
    for node in g.nodes():
        rows.append({"region": node, label_col: partition.get(node), "coastal": g.nodes[node].get("coastal")})
    return pd.DataFrame(rows)


def modularity(g: nx.Graph, partition: dict) -> float:
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, set()).add(node)
    return nx.algorithms.community.modularity(g, communities.values(), weight="weight")


def evaluate_against_coastal_label(df: pd.DataFrame, label_col: str = "community") -> float:
    """Normalized mutual information between detected communities and the
    coastal/continental ground-truth flag, as an external validation check.
    """
    valid = df.dropna(subset=["coastal"])
    if valid.empty:
        logger.warning("No coastal labels available; skipping NMI evaluation.")
        return float("nan")
    coastal_int = valid["coastal"].astype(int)
    return normalized_mutual_info_score(coastal_int, valid[label_col])
