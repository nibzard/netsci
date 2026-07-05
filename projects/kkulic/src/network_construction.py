"""Build the Croatian tourism similarity network.

Nodes are towns/municipalities. Edges connect municipalities whose seasonal
tourism profile (the 12-month normalized distribution of overnight stays,
see ``preprocessing.build_seasonality_fingerprints``) is similar, measured
by cosine similarity. This models destinations that compete for / attract
the same kind of demand (e.g. summer-peaked coastal resorts vs. flat
year-round continental/business destinations), which is the standard
"destination similarity network" construction used in tourism network
research when no direct origin-destination flow matrix is available at the
required spatial granularity.

A graph is built per year (temporal network, see ``dynamics.py``) and a
pooled multi-year graph (mean fingerprint across years) is built as the
primary graph used by the rest of the analysis.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    EDGE_SIMILARITY_THRESHOLD,
    EDGE_TABLE,
    EDGE_TOP_K,
    GRAPH_FILE,
    MONTHS,
    NODE_TABLE,
    SEASONALITY_TABLE,
    TEMPORAL_GRAPH_DIR,
)

logger = logging.getLogger(__name__)


def _similarity_edges(
    region_ids: list[str], matrix: np.ndarray, threshold: float, top_k: int | None
) -> pd.DataFrame:
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, -1.0)

    edges = []
    n = len(region_ids)
    for i in range(n):
        if top_k:
            candidate_idx = np.argsort(sim[i])[::-1][:top_k]
        else:
            candidate_idx = np.where(sim[i] >= threshold)[0]
        for j in candidate_idx:
            if sim[i, j] < threshold and not top_k:
                continue
            if i == j:
                continue
            a, b = sorted((region_ids[i], region_ids[j]))
            edges.append((a, b, float(sim[i, j])))

    edge_df = pd.DataFrame(edges, columns=["source", "target", "weight"]).drop_duplicates(
        subset=["source", "target"]
    )
    return edge_df.reset_index(drop=True)


def build_graph_for_fingerprints(
    fingerprints: pd.DataFrame,
    threshold: float = EDGE_SIMILARITY_THRESHOLD,
    top_k: int | None = EDGE_TOP_K,
) -> nx.Graph:
    region_ids = fingerprints["region"].tolist()
    matrix = fingerprints[MONTHS].to_numpy()
    edge_df = _similarity_edges(region_ids, matrix, threshold, top_k)

    g = nx.Graph()
    g.add_nodes_from(region_ids)
    for _, row in edge_df.iterrows():
        g.add_edge(row["source"], row["target"], weight=row["weight"])
    return g


def attach_node_attributes(g: nx.Graph, nodes: pd.DataFrame) -> nx.Graph:
    nodes_indexed = nodes.set_index("region")
    for col in nodes_indexed.columns:
        values = nodes_indexed[col].to_dict()
        nx.set_node_attributes(g, {n: values.get(n) for n in g.nodes}, name=col)
    return g


def build_pooled_graph(
    fingerprints: pd.DataFrame, nodes: pd.DataFrame
) -> tuple[nx.Graph, pd.DataFrame]:
    """Average fingerprints across all available years per region, then build one graph."""
    weighted = fingerprints.copy()
    weight_col = "total_nights"
    pooled_rows = []
    for region, grp in weighted.groupby("region"):
        w = grp[weight_col].to_numpy()
        w_sum = w.sum()
        if w_sum == 0:
            avg = grp[MONTHS].mean().to_numpy()
        else:
            avg = (grp[MONTHS].to_numpy() * w[:, None]).sum(axis=0) / w_sum
        pooled_rows.append([region, *avg, grp[weight_col].sum()])
    pooled = pd.DataFrame(pooled_rows, columns=["region", *MONTHS, "total_nights"])

    g = build_graph_for_fingerprints(pooled)
    g = attach_node_attributes(g, nodes)
    nx.set_node_attributes(g, pooled.set_index("region")["total_nights"].to_dict(), name="total_nights")
    return g, pooled


def build_temporal_graphs(fingerprints: pd.DataFrame, nodes: pd.DataFrame) -> dict[int, nx.Graph]:
    """One similarity graph per year, for the network-dynamics analysis."""
    graphs = {}
    for year, grp in fingerprints.groupby("year"):
        grp = grp.reset_index(drop=True)
        g = build_graph_for_fingerprints(grp)
        g = attach_node_attributes(g, nodes)
        nx.set_node_attributes(g, grp.set_index("region")["total_nights"].to_dict(), name="total_nights")
        graphs[int(year)] = g
    return graphs


def save_graph(g: nx.Graph, path=GRAPH_FILE) -> None:
    g_clean = g.copy()
    for _, data in g_clean.nodes(data=True):
        for k, v in list(data.items()):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                data[k] = ""
    nx.write_graphml(g_clean, path)
    logger.info("Saved graph (%d nodes, %d edges) to %s", g.number_of_nodes(), g.number_of_edges(), path)


def run() -> nx.Graph:
    logging.basicConfig(level=logging.INFO)
    nodes = pd.read_parquet(NODE_TABLE)
    fingerprints = pd.read_parquet(SEASONALITY_TABLE)

    g, pooled = build_pooled_graph(fingerprints, nodes)
    save_graph(g)

    edge_df = nx.to_pandas_edgelist(g)
    edge_df.to_parquet(EDGE_TABLE, index=False)

    temporal = build_temporal_graphs(fingerprints, nodes)
    TEMPORAL_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    for year, tg in temporal.items():
        save_graph(tg, TEMPORAL_GRAPH_DIR / f"graph_{year}.graphml")

    return g


if __name__ == "__main__":
    run()
