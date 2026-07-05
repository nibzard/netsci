from src.network_construction import attach_node_attributes, build_graph_for_fingerprints, build_pooled_graph
from src.preprocessing import build_node_table, build_seasonality_fingerprints


def test_pooled_graph_nodes_match_regions(synthetic_long_df):
    nodes = build_node_table(synthetic_long_df)
    fp = build_seasonality_fingerprints(synthetic_long_df)
    g, pooled = build_pooled_graph(fp, nodes)
    assert set(g.nodes()) == set(synthetic_long_df["region"].unique())


def test_pooled_graph_separates_coastal_continental(synthetic_long_df):
    """Coastal resorts (sharp summer peak) should be more similar to each
    other than to flat-profile continental cities, given the synthetic
    seasonal curves in conftest.py.
    """
    nodes = build_node_table(synthetic_long_df)
    fp = build_seasonality_fingerprints(synthetic_long_df)
    # Use a small top_k (smaller than the 10-node test fixture's group sizes)
    # so the graph is sparse enough for the within/cross comparison to be
    # meaningful, rather than near-complete.
    g = build_graph_for_fingerprints(fp.groupby("region").mean(numeric_only=True).reset_index(), top_k=3)
    g = attach_node_attributes(g, nodes)

    coastal_nodes = {n for n, d in g.nodes(data=True) if d.get("coastal")}
    continental_nodes = set(g.nodes()) - coastal_nodes

    cross_edges = sum(1 for u, v in g.edges() if (u in coastal_nodes) != (v in coastal_nodes))
    within_edges = g.number_of_edges() - cross_edges
    assert within_edges >= cross_edges
