"""Shared plotting helpers, saved consistently to reports/figures/."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.config import FIGURES

plt.rcParams["figure.dpi"] = 110


def _savefig(fig, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_network(g: nx.Graph, partition: dict | None = None, name: str = "network.png", seed: int = 42):
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(g, weight="weight", seed=seed, k=0.6 / max(1, g.number_of_nodes()) ** 0.5)
    colors = None
    if partition:
        colors = [partition.get(n, 0) for n in g.nodes()]
    sizes = [80 + 400 * g.degree(n, weight="weight") / max(1, max(dict(g.degree(weight="weight")).values())) for n in g.nodes()]
    nx.draw_networkx_edges(g, pos, alpha=0.15, ax=ax)
    nx.draw_networkx_nodes(g, pos, node_size=sizes, node_color=colors, cmap=plt.cm.tab20, ax=ax)
    ax.set_axis_off()
    _savefig(fig, name)


def plot_degree_distribution(g: nx.Graph, name: str = "degree_distribution.png"):
    degrees = np.array([d for _, d in g.degree()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(degrees, bins=30)
    axes[0].set_title("Degree distribution")
    axes[0].set_xlabel("degree")
    axes[0].set_ylabel("count")

    deg, counts = np.unique(degrees[degrees > 0], return_counts=True)
    axes[1].loglog(deg, counts, "o")
    axes[1].set_title("Log-log degree distribution")
    axes[1].set_xlabel("degree (log)")
    axes[1].set_ylabel("count (log)")
    _savefig(fig, name)


def plot_centrality_bar(df: pd.DataFrame, metric: str, n: int = 15, name: str | None = None):
    top = df.sort_values(metric, ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["region"][::-1], top[metric][::-1])
    ax.set_xlabel(metric)
    ax.set_title(f"Top {n} municipalities by {metric}")
    _savefig(fig, name or f"top_{metric}.png")


def plot_resilience_curves(random_df: pd.DataFrame, targeted_df: pd.DataFrame, name: str = "resilience.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(random_df["removed_fraction"], random_df["giant_component_fraction_mean"], label="random failure")
    ax.fill_between(
        random_df["removed_fraction"],
        random_df["giant_component_fraction_mean"] - random_df["giant_component_fraction_std"],
        random_df["giant_component_fraction_mean"] + random_df["giant_component_fraction_std"],
        alpha=0.2,
    )
    ax.plot(targeted_df["removed_fraction"], targeted_df["giant_component_fraction"], label="targeted attack")
    ax.set_xlabel("fraction of nodes removed")
    ax.set_ylabel("giant component fraction")
    ax.legend()
    _savefig(fig, name)


def plot_temporal_summary(df: pd.DataFrame, name: str = "temporal_summary.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    df.plot(x="year", y="n_edges", ax=axes[0, 0], legend=False, marker="o")
    axes[0, 0].set_title("Edges per year")
    df.plot(x="year", y="density", ax=axes[0, 1], legend=False, marker="o")
    axes[0, 1].set_title("Density per year")
    df.plot(x="year", y="modularity", ax=axes[1, 0], legend=False, marker="o")
    axes[1, 0].set_title("Louvain modularity per year")
    df.plot(x="year", y="avg_clustering", ax=axes[1, 1], legend=False, marker="o")
    axes[1, 1].set_title("Average clustering per year")
    fig.tight_layout()
    _savefig(fig, name)
