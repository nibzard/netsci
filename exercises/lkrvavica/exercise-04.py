import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="Exercise 04 — Gowalla Connectivity & Resilience",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## ① Load Graph (same sample as Exercises 02 & 03)
    """)
    return


@app.cell
def _():
    import kagglehub
    import os
    import random
    import pandas as pd
    import networkx as nx

    path = kagglehub.dataset_download("marquis03/gowalla")
    edge_file = os.path.join(path, "Gowalla_edges.txt")
    df_edges = pd.read_csv(edge_file, sep="\t", header=None, names=["user_a", "user_b"])

    G_full = nx.from_pandas_edgelist(df_edges, source="user_a", target="user_b")
    print(f"Full graph — Nodes: {G_full.number_of_nodes():,}  |  Edges: {G_full.number_of_edges():,}")

    random.seed(42)
    degrees_full = dict(G_full.degree())
    top_node = max(degrees_full, key=lambda n: degrees_full[n])
    bfs_nodes = list(nx.bfs_tree(G_full, top_node).nodes())[:2000]
    G = G_full.subgraph(bfs_nodes).copy()

    # Note: undirected graph — only connected components (no weak/strong split)
    print(f"Sample     — Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    print(f"Graph type: undirected — connected components only (no weak/strong split)")
    return G, nx


@app.cell
def _(mo):
    mo.md("""
    ## ② Connected Components — Before Any Removal
    """)
    return


@app.cell
def _(G, nx):
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_components = len(components)
    sizes = [len(c) for c in components]

    print(f"Number of connected components: {n_components}")
    print(f"Largest component size:         {sizes[0]:,} nodes")
    if len(sizes) > 1:
        print(f"Second largest:                 {sizes[1]:,} nodes")
        print(f"All component sizes:            {sizes}")
    else:
        print("Graph is fully connected — one single component.")

    lcc_frac = sizes[0] / G.number_of_nodes()
    print(f"LCC covers {lcc_frac:.1%} of all sample nodes.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## ③ Articulation Points
    """)
    return


@app.cell
def _(G, nx):
    art_points = list(nx.articulation_points(G))
    art_points_sorted = sorted(art_points, key=lambda n: G.degree(n), reverse=True)

    print(f"Total articulation points found: {len(art_points):,}")
    print()
    if art_points_sorted:
        print("Top 15 articulation points by degree:")
        print(f"{'Node':>8}  {'Degree':>8}")
        print("-" * 20)
        for _n in art_points_sorted[:15]:
            print(f"{_n:>8}  {G.degree(_n):>8}")
    else:
        print("No articulation points — graph is 2-edge-connected (biconnected).")
    return art_points, art_points_sorted


@app.cell
def _(art_points, art_points_sorted, mo):
    top_ap = art_points_sorted[:5] if art_points_sorted else []
    mo.md(f"""
    ### Articulation Points Summary

    An **articulation point** (cut vertex) is a node whose removal increases the number of
    connected components — i.e. it is the only path between some parts of the graph.

    - Total found: **{len(art_points):,}**
    - Top 5 by degree: {", ".join(f"**{n}**" for n in top_ap) if top_ap else "none"}

    > A high number of articulation points means the network has many single points of failure.
    > A high-degree articulation point is especially dangerous — removing it disconnects many nodes at once.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ④ Bridges (Critical Edges)
    """)
    return


@app.cell
def _(G, nx):
    bridges = list(nx.bridges(G))
    bridges_sorted = sorted(bridges, key=lambda e: G.degree(e[0]) + G.degree(e[1]), reverse=True)

    print(f"Total bridges found: {len(bridges):,}")
    print()
    if bridges_sorted:
        print("Top 15 bridges by combined endpoint degree:")
        print(f"{'Edge':>20}  {'Deg(u)':>8}  {'Deg(v)':>8}  {'Combined':>10}")
        print("-" * 50)
        for _u, _v in bridges_sorted[:15]:
            print(f"({_u:>5}, {_v:>5})        {G.degree(_u):>8}  {G.degree(_v):>8}  {G.degree(_u)+G.degree(_v):>10}")
    else:
        print("No bridges — every edge belongs to a cycle (graph is 2-edge-connected).")
    return bridges, bridges_sorted


@app.cell
def _(bridges, bridges_sorted, mo):
    top_br = bridges_sorted[:3] if bridges_sorted else []
    mo.md(f"""
    ### Bridges Summary

    A **bridge** is an edge whose removal disconnects the graph (or increases component count).
    It is the only connection between two parts of the network.

    - Total bridges found: **{len(bridges):,}**
    - Top bridges by combined endpoint degree: {", ".join(f"({u}↔{v})" for u,v in top_br) if top_br else "none"}

    > Many bridges indicate a network that is structurally fragile along specific edges.
    > Zero bridges means the graph is 2-edge-connected — every connection has at least one backup route.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑤ Node Removal — Remove the Top Articulation Point
    """)
    return


@app.cell
def _(G, art_points_sorted, nx):
    # Before state
    before_nodes = G.number_of_nodes()
    before_edges = G.number_of_edges()
    before_components = nx.number_connected_components(G)
    before_lcc = len(max(nx.connected_components(G), key=len))

    # Choose critical node: highest-degree articulation point
    # If no articulation points, fall back to highest-degree node
    if art_points_sorted:
        critical_node = art_points_sorted[0]
        removal_reason = "highest-degree articulation point"
    else:
        critical_node = max(G.nodes(), key=lambda n: G.degree(n))
        removal_reason = "highest-degree node (no articulation points found)"

    print(f"Removing node {critical_node} ({removal_reason}, degree={G.degree(critical_node)})")

    # After removal
    G_no_node = G.copy()
    G_no_node.remove_node(critical_node)

    after_nodes = G_no_node.number_of_nodes()
    after_edges = G_no_node.number_of_edges()
    after_components = nx.number_connected_components(G_no_node)
    after_lcc = len(max(nx.connected_components(G_no_node), key=len))

    delta_components = after_components - before_components
    delta_lcc = after_lcc - (before_lcc - 1)  # adjust for the removed node itself

    print()
    print(f"{'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 62)
    print(f"{'Nodes':<30} {before_nodes:>10} {after_nodes:>10} {after_nodes - before_nodes:>+10}")
    print(f"{'Edges':<30} {before_edges:>10} {after_edges:>10} {after_edges - before_edges:>+10}")
    print(f"{'Connected components':<30} {before_components:>10} {after_components:>10} {delta_components:>+10}")
    print(f"{'LCC size':<30} {before_lcc:>10} {after_lcc:>10} {after_lcc - before_lcc:>+10}")
    print(f"{'LCC % of remaining':<30} {after_lcc/after_nodes:.1%}")
    return (
        G_no_node,
        after_components,
        after_edges,
        after_lcc,
        after_nodes,
        before_components,
        before_edges,
        before_lcc,
        before_nodes,
        critical_node,
        removal_reason,
    )


@app.cell
def _(mo):
    mo.md("""
    ## ⑥ Edge Removal — Remove the Top Bridge
    """)
    return


@app.cell
def _(G, bridges_sorted, nx):
    before_comp_edge = nx.number_connected_components(G)
    before_lcc_edge = len(max(nx.connected_components(G), key=len))

    if bridges_sorted:
        critical_edge = bridges_sorted[0]
        edge_reason = "highest combined-degree bridge"
    else:
        # Fall back: remove edge between two highest-degree nodes
        deg_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        _eu, _ev = deg_sorted[0][0], deg_sorted[1][0]
        critical_edge = (_eu, _ev) if G.has_edge(_eu, _ev) else list(G.edges(deg_sorted[0][0]))[0]
        edge_reason = "edge between two highest-degree nodes (no bridges found)"

    print(f"Removing edge {critical_edge} ({edge_reason})")

    G_no_edge = G.copy()
    G_no_edge.remove_edge(*critical_edge)

    after_comp_edge = nx.number_connected_components(G_no_edge)
    after_lcc_edge = len(max(nx.connected_components(G_no_edge), key=len))

    print()
    print(f"{'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 62)
    print(f"{'Connected components':<30} {before_comp_edge:>10} {after_comp_edge:>10} {after_comp_edge - before_comp_edge:>+10}")
    print(f"{'LCC size':<30} {before_lcc_edge:>10} {after_lcc_edge:>10} {after_lcc_edge - before_lcc_edge:>+10}")
    return after_comp_edge, after_lcc_edge, before_comp_edge, critical_edge


@app.cell
def _(mo):
    mo.md("""
    ## ⑦ Before / After Summary Table
    """)
    return


@app.cell
def _(
    G,
    after_comp_edge,
    after_components,
    after_edges,
    after_lcc,
    after_lcc_edge,
    after_nodes,
    before_components,
    before_edges,
    before_lcc,
    before_nodes,
    critical_edge,
    critical_node,
    mo,
):
    lcc_pct_before = before_lcc / before_nodes
    lcc_pct_after_node = after_lcc / after_nodes
    lcc_pct_after_edge = after_lcc_edge / G.number_of_nodes()

    mo.md(f"""
    ### Before / After Removal Summary

    | Metric | Original | After node {critical_node} removed | After edge {critical_edge} removed |
    |---|---|---|---|
    | Nodes | {before_nodes:,} | {after_nodes:,} | {before_nodes:,} |
    | Edges | {before_edges:,} | {after_edges:,} | {before_edges - 1:,} |
    | Components | {before_components} | {after_components} | {after_comp_edge} |
    | LCC size | {before_lcc:,} | {after_lcc:,} | {after_lcc_edge:,} |
    | LCC % of graph | {lcc_pct_before:.1%} | {lcc_pct_after_node:.1%} | {lcc_pct_after_edge:.1%} |
    """)
    return lcc_pct_after_edge, lcc_pct_after_node, lcc_pct_before


@app.cell
def _(mo):
    mo.md("""
    ## ⑧ Visualization — Before and After Node Removal
    """)
    return


@app.cell
def _(G, G_no_node, art_points, bridges, critical_node, mo, nx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import random as _r

    _r.seed(42)

    # Use a 300-node ego sample for readable viz
    hub = critical_node
    ego_before = nx.ego_graph(G, hub, radius=2)
    if ego_before.number_of_nodes() > 300:
        _keep = [hub] + _r.sample(list(ego_before.nodes() - {hub}), 299)
        ego_before = ego_before.subgraph(_keep).copy()

    # After: same nodes minus the removed hub
    ego_after_nodes = [n for n in ego_before.nodes() if n != hub]
    ego_after = G_no_node.subgraph(ego_after_nodes).copy()

    # Shared layout based on before-graph nodes (excluding hub for after)
    pos_before = nx.spring_layout(ego_before, seed=42, k=0.45)
    pos_after = {n: pos_before[n] for n in ego_after.nodes()}

    art_set = set(art_points)
    bridge_set = set(map(frozenset, bridges))

    def node_color(n, removed=False):
        if removed:
            return "#ff4444"
        if n in art_set:
            return "#f7c948"   # gold = articulation point
        return "#4f8ef7"       # blue = normal

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#090b14")

    for _ax in axes:
        _ax.set_facecolor("#090b14")
        _ax.axis("off")

    # --- LEFT: Before ---
    colors_before = [node_color(n) for n in ego_before.nodes()]
    sizes_before = [40 + G.degree(n) * 2.5 for n in ego_before.nodes()]

    edge_colors_before = []
    for _u, _v in ego_before.edges():
        if frozenset([_u, _v]) in bridge_set:
            edge_colors_before.append("#ff6b35")   # orange = bridge
        else:
            edge_colors_before.append("#3a4a6b")

    nx.draw_networkx_edges(ego_before, pos_before, ax=axes[0],
                           edge_color=edge_colors_before, width=0.8, alpha=0.7)
    nx.draw_networkx_nodes(ego_before, pos_before, ax=axes[0],
                           node_color=colors_before, node_size=sizes_before, alpha=0.9)
    nx.draw_networkx_labels(ego_before, pos_before, ax=axes[0],
                            labels={critical_node: str(critical_node)},
                            font_color="white", font_size=8, font_weight="bold")
    axes[0].set_title(
        f"BEFORE — {ego_before.number_of_nodes()} nodes, {ego_before.number_of_edges()} edges\n"
        f"🟡 gold = articulation point  🟠 orange edge = bridge",
        color="#dde0f0", fontsize=11, pad=10)

    # --- RIGHT: After ---
    after_comps = list(nx.connected_components(ego_after))
    comp_palette = ["#4f8ef7", "#f76f8e", "#50e3c2", "#f7c948", "#c084fc", "#fb923c"]
    node_comp_color = {}
    for _idx, _comp in enumerate(sorted(after_comps, key=len, reverse=True)):
        for _n in _comp:
            node_comp_color[_n] = comp_palette[_idx % len(comp_palette)]

    colors_after = [node_comp_color.get(n, "#888") for n in ego_after.nodes()]
    sizes_after = [40 + G_no_node.degree(n) * 2.5 for n in ego_after.nodes()]

    nx.draw_networkx_edges(ego_after, pos_after, ax=axes[1],
                           edge_color="#3a4a6b", width=0.8, alpha=0.7)
    nx.draw_networkx_nodes(ego_after, pos_after, ax=axes[1],
                           node_color=colors_after, node_size=sizes_after, alpha=0.9)
    axes[1].set_title(
        f"AFTER removing node {critical_node}\n"
        f"{ego_after.number_of_nodes()} nodes, {ego_after.number_of_edges()} edges\n"
        f"Each color = separate connected component",
        color="#dde0f0", fontsize=11, pad=10)

    plt.suptitle("Gowalla — Connectivity Before & After Critical Node Removal",
                 color="#e0e4f5", fontsize=13, y=1.01)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑨ Interpretation & Resilience Note
    """)
    return


@app.cell
def _(
    after_comp_edge,
    after_components,
    after_lcc,
    after_lcc_edge,
    art_points,
    before_comp_edge,
    before_lcc,
    bridges,
    critical_edge,
    critical_node,
    lcc_pct_after_edge,
    lcc_pct_after_node,
    lcc_pct_before,
    mo,
    removal_reason,
):
    node_delta_comp = after_components - 1   # components added by removal
    edge_delta_comp = after_comp_edge - before_comp_edge
    lcc_drop_node = before_lcc - after_lcc
    lcc_drop_edge = before_lcc - after_lcc_edge

    mo.md(f"""
    ### 📝 Method Note
    **Commands used:** `nx.connected_components`, `nx.number_connected_components`,
    `nx.articulation_points`, `nx.bridges`, `nx.biconnected_components`,
    `G.remove_node`, `G.remove_edge`, manual before/after comparison, `nx.spring_layout`

    ---

    ### 📋 Component Counts — Before and After

    | Scenario | Components | LCC size | LCC % |
    |---|---|---|---|
    | Original graph | 1 | {before_lcc:,} | {lcc_pct_before:.1%} |
    | After removing node **{critical_node}** | {after_components} | {after_lcc:,} | {lcc_pct_after_node:.1%} |
    | After removing edge **{critical_edge}** | {after_comp_edge} | {after_lcc_edge:,} | {lcc_pct_after_edge:.1%} |

    ---

    ### 📋 Articulation Points & Bridges

    | Structural feature | Count |
    |---|---|
    | Articulation points | **{len(art_points):,}** |
    | Bridges | **{len(bridges):,}** |

    ---

    ### 🧭 Resilience Note

    **Articulation points ({len(art_points):,} found):** The sample contains {len(art_points):,} nodes
    whose removal would increase the number of connected components.
    {"This is a significant number relative to the 2,000-node sample — roughly " +
    f"{len(art_points)/2000:.1%} of all nodes are single points of failure." if len(art_points) > 0
    else "The absence of articulation points means the graph is biconnected — every node has at least two independent paths to every other node, giving strong resilience."}

    **Critical node removal (node {critical_node}, {removal_reason}):**
    Removing it {"increased" if node_delta_comp > 0 else "did not increase"} the component count
    {"by " + str(node_delta_comp) + " — the LCC shrank from " + str(before_lcc) + " to " + str(after_lcc) +
    f" nodes ({lcc_drop_node:,} nodes lost from the main component)." if node_delta_comp > 0
    else f"— the graph remained connected, dropping only {lcc_drop_node} nodes from the LCC."}
    {"This is severe fragmentation: a single user's removal broke the network into multiple components." if node_delta_comp > 5
    else "This shows the node was critical for routing but the network did not shatter completely." if node_delta_comp > 0
    else "This shows the network has structural redundancy — backup paths exist even without this node."}

    **Critical edge removal (edge {critical_edge}):**
    {"Removing this bridge split the graph — component count went from " + str(before_comp_edge) +
    " to " + str(after_comp_edge) + " and LCC shrank by " + str(lcc_drop_edge) + " nodes."
    if edge_delta_comp > 0
    else "Removing this edge did not disconnect the graph — the two endpoints had alternative paths between them."}

    **Overall resilience assessment:**
    {"The network shows **moderate to high vulnerability**. With " + str(len(art_points)) +
    " articulation points and " + str(len(bridges)) + " bridges, there are many single points of failure. "
    "This is typical of a network grown around a dominant hub (node 307): the hub and its immediate connectors "
    "become structurally essential, while peripheral nodes are only weakly tied in. "
    "In Gowalla terms, this means local friendship neighbourhoods are indeed connected *through* a small set "
    "of high-degree users — remove those users and multiple friend groups lose their connection to the rest "
    "of the network. The network lacks redundancy at its periphery." if len(art_points) > 10
    else "The network shows **moderate resilience**. Articulation points exist but are limited, "
    "suggesting some redundant paths. The hub node (307) is still critical for global routing "
    "but the neighbourhood structure is not entirely dependent on single connectors."
    if len(art_points) > 0
    else "The network shows **strong resilience** — no articulation points means every node has "
    "multiple independent paths to every other node. Even removing the highest-degree node "
    "does not fragment the graph. This is a biconnected network with genuine redundancy."}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
