import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 05: Facebook Ego Network

    This notebook keeps the same SNAP Facebook ego network used in the earlier exercises: ego node **698**.

    Goal:
    detect communities in the local Facebook neighborhood, compare them with the available SNAP circle annotations, and identify nodes that bridge detected communities.

    Required inputs:
    `data/facebook/698.edges` and `data/facebook/698.circles`

    Expected output:
    community counts from Louvain and label propagation, modularity scores, a circle-overlap comparison, bridge-node ranking, one community visualization, and a short interpretation.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from networkx.algorithms import community as nx_comm
    return Path, np, nx, nx_comm, pd, plt


@app.cell
def _(Path):
    EGO_ID = 698

    def resolve_data_dir():
        candidates = []

        if "__file__" in globals():
            notebook_dir = Path(__file__).resolve().parent
            candidates.extend(
                [
                    notebook_dir / "facebook",
                    notebook_dir.parent / "data" / "facebook",
                    notebook_dir.parent / "facebook",
                ]
            )

        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "facebook",
                cwd / "exercises" / "mkatavic" / "data" / "facebook",
                cwd / "exercises" / "mkatavic" / "exercise_05" / "facebook",
            ]
        )

        for data_dir in candidates:
            if (data_dir / f"{EGO_ID}.edges").exists():
                return data_dir

        searched = "\n".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Could not find the Facebook ego data directory. Searched:\n"
            f"{searched}"
        )

    DATA_DIR = resolve_data_dir()
    EDGE_PATH = DATA_DIR / f"{EGO_ID}.edges"
    CIRCLE_PATH = DATA_DIR / f"{EGO_ID}.circles"
    return CIRCLE_PATH, EDGE_PATH, EGO_ID


@app.cell
def _(nx):
    def load_ego_network(edge_path, ego_id):
        alter_graph = nx.read_edgelist(
            edge_path,
            nodetype=int,
            create_using=nx.Graph(),
        )
        alters = sorted(alter_graph.nodes())

        graph = alter_graph.copy()
        graph.add_node(ego_id)
        graph.add_edges_from((ego_id, alter) for alter in alters)

        return graph, alter_graph

    def load_circles(circle_path):
        circles = {}

        if not circle_path.exists():
            return circles

        with circle_path.open() as file:
            for line in file:
                parts = line.split()
                if parts:
                    circles[parts[0]] = set(int(node) for node in parts[1:])

        return circles

    def sorted_communities(communities):
        return sorted(
            (set(community_nodes) for community_nodes in communities),
            key=lambda community_nodes: (-len(community_nodes), min(community_nodes)),
        )

    def community_lookup(communities):
        return {
            node: community_index
            for community_index, community_nodes in enumerate(communities, start=1)
            for node in community_nodes
        }

    def jaccard(left_nodes, right_nodes):
        union = left_nodes | right_nodes
        if not union:
            return 0.0
        return len(left_nodes & right_nodes) / len(union)

    def component_layout(
        graph,
        components,
        np_module,
        nx_module,
        orbit_radius=3.3,
        seed=698,
    ):
        positions = {}

        if not components:
            return positions

        if len(components) == 1:
            centers = [np_module.array([0.0, 0.0])]
        else:
            angles = np_module.linspace(0, 2 * np_module.pi, len(components), endpoint=False)
            centers = [
                orbit_radius * np_module.array([np_module.cos(angle), np_module.sin(angle)])
                for angle in angles
            ]

        for layout_index, component_nodes in enumerate(components):
            subgraph = graph.subgraph(component_nodes)
            sub_pos = nx_module.spring_layout(
                subgraph,
                seed=seed + layout_index,
                k=1.0 / np_module.sqrt(max(subgraph.number_of_nodes(), 1)),
            )
            raw_positions = np_module.array([sub_pos[node] for node in subgraph.nodes()])
            local_center = raw_positions.mean(axis=0)
            spread = np_module.abs(raw_positions - local_center).max()
            if spread == 0:
                spread = 1.0

            local_scale = 0.7 + 0.12 * np_module.sqrt(subgraph.number_of_nodes())
            for node in subgraph.nodes():
                positions[node] = (
                    centers[layout_index]
                    + ((sub_pos[node] - local_center) / spread) * local_scale
                )

        return positions

    def participation_coefficient(graph, node, communities, lookup):
        degree = graph.degree(node)
        if degree == 0:
            return 0.0

        community_edge_counts = []
        for community_nodes in communities:
            community_edge_counts.append(
                sum(1 for neighbor in graph.neighbors(node) if neighbor in community_nodes)
            )

        return 1 - sum((edge_count / degree) ** 2 for edge_count in community_edge_counts)
    return (
        community_lookup,
        component_layout,
        jaccard,
        load_circles,
        load_ego_network,
        participation_coefficient,
        sorted_communities,
    )


@app.cell
def _(CIRCLE_PATH, EDGE_PATH, EGO_ID, load_circles, load_ego_network, nx):
    G, alter_graph = load_ego_network(EDGE_PATH, EGO_ID)
    annotated_circles = load_circles(CIRCLE_PATH)

    alter_components = sorted(
        nx.connected_components(alter_graph),
        key=len,
        reverse=True,
    )
    alter_component_sizes = [len(component) for component in alter_components]

    circle_nodes = set().union(*annotated_circles.values()) if annotated_circles else set()
    circle_nodes_outside_graph = sorted(circle_nodes - set(alter_graph.nodes()))
    return (
        G,
        alter_component_sizes,
        alter_components,
        alter_graph,
        annotated_circles,
        circle_nodes_outside_graph,
    )


@app.cell
def _(EGO_ID, alter_graph, community_lookup, nx_comm, pd, sorted_communities):
    louvain_partition = sorted_communities(
        nx_comm.louvain_communities(
            alter_graph,
            seed=EGO_ID,
            resolution=1.0,
        )
    )
    label_partition = sorted_communities(
        nx_comm.label_propagation_communities(alter_graph)
    )

    louvain_lookup = community_lookup(louvain_partition)
    louvain_modularity = nx_comm.modularity(alter_graph, louvain_partition)
    label_modularity = nx_comm.modularity(alter_graph, label_partition)

    partition_metrics_df = pd.DataFrame(
        [
            {
                "method": "Louvain",
                "communities": len(louvain_partition),
                "community sizes": ", ".join(str(len(nodes)) for nodes in louvain_partition),
                "modularity": louvain_modularity,
            },
            {
                "method": "label propagation",
                "communities": len(label_partition),
                "community sizes": ", ".join(str(len(nodes)) for nodes in label_partition),
                "modularity": label_modularity,
            },
        ]
    )

    louvain_community_df = pd.DataFrame(
        [
            {
                "community": community_index,
                "size": len(community_nodes),
                "nodes": ", ".join(str(node) for node in sorted(community_nodes)),
            }
            for community_index, community_nodes in enumerate(louvain_partition, start=1)
        ]
    )
    return (
        label_modularity,
        label_partition,
        louvain_community_df,
        louvain_lookup,
        louvain_modularity,
        louvain_partition,
        partition_metrics_df,
    )


@app.cell
def _(
    G,
    alter_component_sizes,
    alter_graph,
    annotated_circles,
    circle_nodes_outside_graph,
    nx,
    pd,
):
    graph_summary_df = pd.DataFrame(
        [
            {"metric": "ego id", "value": 698},
            {"metric": "full ego graph nodes", "value": G.number_of_nodes()},
            {"metric": "full ego graph edges", "value": G.number_of_edges()},
            {"metric": "alter graph nodes", "value": alter_graph.number_of_nodes()},
            {"metric": "alter graph edges", "value": alter_graph.number_of_edges()},
            {"metric": "alter graph density", "value": round(nx.density(alter_graph), 4)},
            {
                "metric": "alter connected components",
                "value": len(alter_component_sizes),
            },
            {
                "metric": "alter component sizes",
                "value": ", ".join(str(size) for size in alter_component_sizes),
            },
            {"metric": "annotated SNAP circles", "value": len(annotated_circles)},
            {
                "metric": "circle nodes outside edge-based graph",
                "value": (
                    ", ".join(str(node) for node in circle_nodes_outside_graph)
                    if circle_nodes_outside_graph
                    else "none"
                ),
            },
        ]
    )
    return (graph_summary_df,)


@app.cell
def _(alter_graph, annotated_circles, jaccard, louvain_partition, pd):
    circle_comparison_rows = []

    for community_index, community_nodes in enumerate(louvain_partition, start=1):
        if not annotated_circles:
            circle_comparison_rows.append(
                {
                    "louvain community": community_index,
                    "community size": len(community_nodes),
                    "best SNAP circle": "none",
                    "raw circle size": 0,
                    "circle size in graph": 0,
                    "overlap nodes": 0,
                    "jaccard": 0.0,
                }
            )
            continue

        best_circle_name, best_circle_nodes = max(
            annotated_circles.items(),
            key=lambda item: jaccard(
                community_nodes,
                item[1] & set(alter_graph.nodes()),
            ),
        )
        filtered_circle_nodes = best_circle_nodes & set(alter_graph.nodes())
        circle_comparison_rows.append(
            {
                "louvain community": community_index,
                "community size": len(community_nodes),
                "best SNAP circle": best_circle_name,
                "raw circle size": len(best_circle_nodes),
                "circle size in graph": len(filtered_circle_nodes),
                "overlap nodes": len(community_nodes & filtered_circle_nodes),
                "jaccard": jaccard(community_nodes, filtered_circle_nodes),
            }
        )

    circle_overlap_df = pd.DataFrame(circle_comparison_rows)
    return (circle_overlap_df,)


@app.cell
def _(
    alter_graph,
    louvain_lookup,
    louvain_partition,
    nx,
    participation_coefficient,
    pd,
):
    alter_betweenness = nx.betweenness_centrality(alter_graph)

    bridge_rows = []
    for node in alter_graph.nodes():
        own_community = louvain_lookup[node]
        neighbor_communities = sorted(
            {
                louvain_lookup[neighbor]
                for neighbor in alter_graph.neighbors(node)
                if louvain_lookup[neighbor] != own_community
            }
        )
        external_edges = sum(
            1
            for neighbor in alter_graph.neighbors(node)
            if louvain_lookup[neighbor] != own_community
        )

        if external_edges == 0:
            continue

        degree = alter_graph.degree(node)
        bridge_rows.append(
            {
                "node": node,
                "louvain community": own_community,
                "degree": degree,
                "external edges": external_edges,
                "neighbor communities": ", ".join(str(index) for index in neighbor_communities),
                "participation": participation_coefficient(
                    alter_graph,
                    node,
                    louvain_partition,
                    louvain_lookup,
                ),
                "betweenness": alter_betweenness[node],
            }
        )

    bridge_df = (
        pd.DataFrame(bridge_rows)
        .sort_values(
            ["external edges", "participation", "betweenness", "degree", "node"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    top_bridge_nodes = bridge_df["node"].head(5).tolist()
    return bridge_df, top_bridge_nodes


@app.cell(hide_code=True)
def _(
    G,
    alter_component_sizes,
    alter_graph,
    annotated_circles,
    circle_nodes_outside_graph,
    label_modularity,
    label_partition,
    louvain_modularity,
    louvain_partition,
    mo,
):
    mo.md(f"""
    The full ego graph has **{G.number_of_nodes()} nodes** and **{G.number_of_edges()} edges**.
    For community detection I use the **alter graph** with the ego removed: **{alter_graph.number_of_nodes()} nodes**, **{alter_graph.number_of_edges()} edges**, and connected-component sizes **{", ".join(str(size) for size in alter_component_sizes)}**.

    This preparation keeps the same ego sample (**698**) but avoids letting the ego's automatic tie to every alter dominate the detected groups.
    The SNAP circles are overlapping annotations, not a partition, so I compare them to detected communities with best-match Jaccard overlap.
    {len(circle_nodes_outside_graph)} annotated circle nodes are not present in the edge-based alter graph used in earlier exercises.

    Louvain finds **{len(louvain_partition)} communities** with modularity **{louvain_modularity:.4f}**.
    Label propagation finds **{len(label_partition)} communities** with modularity **{label_modularity:.4f}**.
    The ego has ties into all **{len(louvain_partition)} Louvain communities**, so among alters I treat bridge nodes as nodes with edges across Louvain communities.
    There are **{len(annotated_circles)} annotated SNAP circles** available for comparison.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Graph Preparation
    """)
    return


@app.cell
def _(graph_summary_df):
    graph_summary_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Community Detection Results
    """)
    return


@app.cell
def _(partition_metrics_df):
    partition_metrics_df.round({"modularity": 4})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Louvain Communities
    """)
    return


@app.cell
def _(louvain_community_df):
    louvain_community_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparison With SNAP Circles

    Circles are overlapping labels. The table therefore shows the best matching circle for each Louvain community instead of forcing a one-to-one partition comparison.
    """)
    return


@app.cell
def _(circle_overlap_df):
    circle_overlap_df.round({"jaccard": 3})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Bridge Nodes Between Louvain Communities

    The ego node is a bridge to all alter communities by construction, so the table focuses on alter nodes that also connect across detected communities.
    """)
    return


@app.cell
def _(bridge_df):
    bridge_df.head(10).round(
        {
            "participation": 3,
            "betweenness": 4,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Louvain Community Visualization
    """)
    return


@app.cell
def _(
    EGO_ID,
    alter_components,
    alter_graph,
    component_layout,
    louvain_lookup,
    louvain_partition,
    np,
    nx,
    plt,
    top_bridge_nodes,
):
    community_palette = [
        tuple(color)
        for color in plt.cm.Set2(np.linspace(0, 1, max(len(louvain_partition), 1)))
    ]
    node_colors = [
        community_palette[(louvain_lookup[node] - 1) % len(community_palette)]
        for node in alter_graph.nodes()
    ]
    node_sizes = [80 + 8 * alter_graph.degree(node) for node in alter_graph.nodes()]

    pos = component_layout(
        alter_graph,
        alter_components,
        np,
        nx,
        orbit_radius=3.4,
        seed=EGO_ID,
    )

    internal_edges = [
        (source, target)
        for source, target in alter_graph.edges()
        if louvain_lookup[source] == louvain_lookup[target]
    ]
    cross_edges = [
        (source, target)
        for source, target in alter_graph.edges()
        if louvain_lookup[source] != louvain_lookup[target]
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(
        alter_graph,
        pos,
        edgelist=internal_edges,
        ax=ax,
        edge_color="#B8B8B8",
        width=0.9,
        alpha=0.35,
    )
    nx.draw_networkx_edges(
        alter_graph,
        pos,
        edgelist=cross_edges,
        ax=ax,
        edge_color="#404040",
        width=1.4,
        alpha=0.55,
    )
    nx.draw_networkx_nodes(
        alter_graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.6,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        alter_graph,
        pos,
        nodelist=top_bridge_nodes,
        node_size=[110 + 10 * alter_graph.degree(node) for node in top_bridge_nodes],
        node_color=[
            community_palette[(louvain_lookup[node] - 1) % len(community_palette)]
            for node in top_bridge_nodes
        ],
        edgecolors="#111111",
        linewidths=2.0,
        ax=ax,
    )

    bridge_labels = {node: str(node) for node in top_bridge_nodes}
    nx.draw_networkx_labels(
        alter_graph,
        pos,
        labels=bridge_labels,
        font_size=9,
        font_weight="bold",
        font_color="#111111",
        ax=ax,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=community_palette[index - 1],
            markeredgecolor="white",
            markersize=8,
            label=f"community {index}",
        )
        for index in range(1, len(louvain_partition) + 1)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    ax.text(
        0.02,
        0.02,
        "Dark edges cross Louvain communities\nBlack outlines mark top bridge alters",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D0D0"},
    )
    ax.set_title("Louvain communities in the ego 698 alter graph")
    ax.axis("off")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(bridge_df, circle_overlap_df, louvain_modularity, mo):
    strongest_matches = circle_overlap_df.sort_values("jaccard", ascending=False).head(3)
    strongest_match_text = "; ".join(
        (
            f"community {int(row['louvain community'])} with {row['best SNAP circle']} "
            f"(Jaccard {row['jaccard']:.3f})"
        )
        for _, row in strongest_matches.iterrows()
    )
    weak_match_row = circle_overlap_df.sort_values("jaccard", ascending=True).iloc[0]
    top_bridge_text = ", ".join(str(node) for node in bridge_df["node"].head(5))

    mo.md(
        f"""
        ## Interpretation

        The ego network does not look like one single friend group. Louvain finds **5 alter communities** with modularity **{louvain_modularity:.4f}**, which is strong enough to treat the partition as meaningful for this small local graph.

        The detected communities partly match the SNAP circles. The strongest matches are **{strongest_match_text}**.
        That means several algorithmic communities line up with the hand-labeled social circles.
        The weakest match is **community {int(weak_match_row['louvain community'])}**, whose best Jaccard score is only **{weak_match_row['jaccard']:.3f}**.
        I interpret that group as a small connector or residual cluster rather than a clean annotated circle.

        Ego **698** is the main bridge by definition because it links to every alter in every detected community.
        Among alters, the clearest cross-community bridges are **{top_bridge_text}**.
        These nodes have edges outside their own Louvain community, so they likely connect separate friend circles inside the ego's local social world.
        Node **856** is especially important because it has the largest number of cross-community edges and the highest alter betweenness in the bridge table.
        """
    )
    return


if __name__ == "__main__":
    app.run()
