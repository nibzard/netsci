import networkx as nx

from src.centrality import compute_centralities
from src.resilience import simulate_random_failure, simulate_targeted_attack


def _star_like_graph():
    g = nx.barabasi_albert_graph(30, 2, seed=1)
    for u, v in g.edges():
        g[u][v]["weight"] = 1.0
    return g


def test_centrality_dataframe_complete():
    g = _star_like_graph()
    df = compute_centralities(g)
    assert len(df) == g.number_of_nodes()
    for col in ("degree", "strength", "betweenness", "closeness", "eigenvector", "pagerank"):
        assert col in df.columns
        assert df[col].notna().all()


def test_targeted_attack_breaks_giant_component_faster_than_random():
    g = _star_like_graph()
    random_df = simulate_random_failure(g, n_repeats=5)
    targeted_df = simulate_targeted_attack(g, metric="degree", adaptive=True)

    def frac_removed_to_halve(df, col):
        below_half = df[df[col] <= 0.5]
        return below_half["removed_fraction"].iloc[0] if len(below_half) else 1.0

    r_frac = frac_removed_to_halve(random_df, "giant_component_fraction_mean")
    t_frac = frac_removed_to_halve(targeted_df, "giant_component_fraction")
    assert t_frac <= r_frac
