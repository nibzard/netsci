import marimo

__generated_with = "0.0.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import deque
    from pathlib import Path

    import gzip
    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import random

    data_path = Path(__file__).with_name("web-Google.txt.gz")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    return data_path, deque, gzip, mo, nx, plt, random


@app.cell
def _(mo):
    mo.md(
        """
        # Exercise 02
        """
    )
    return


@app.cell
def _():
    # all_nodes = set()
    # all_edges = []
    #
    # with gzip.open('web-Google.txt.gz', 'rt') as f:
    #   for line in f:
    #     if line.startswith('#'):
    #       continue
    #     u, v = map(int, line.split())
    #     all_nodes.add(u)
    #     all_nodes.add(v)
    #     all_edges.append((u, v))
    #
    # sample_nodes = set(random.sample(list(all_nodes), 300))
    #
    # sample_edges = [(u, v) for u, v in all_edges if u in sample_nodes and v in sample_nodes]
    return


@app.cell
def _(mo):
    mo.md(
        """
        The commented code above, on its own, is practically (usually) useless since it uses the above sampling method on a relatively big dataset which won't reallz work considering how small the wanted sample (200-300 nodes) is.
        """
    )
    return


@app.cell
def _(data_path, deque, gzip, random):
    adj = {}

    with gzip.open(data_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = map(int, line.split())
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, [])

    start = random.choice(list(adj.keys()))
    visited = set([start])
    queue = deque([start])

    while queue and len(visited) < 300:
        _node = queue.popleft()
        for neighbor in adj.get(_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
            if len(visited) >= 300:
                break

    # keep edges inside sampled nodes
    sample_edges = [(_u, _v) for _u in visited for _v in adj.get(_u, []) if _v in visited]
    return adj, sample_edges, start, visited


@app.cell
def _(nx, sample_edges, visited):
    G = nx.DiGraph()
    G.add_nodes_from(visited)
    G.add_edges_from(sample_edges)

    print(f"Nodes: {nx.number_of_nodes(G)}, Vertices: {nx.number_of_edges(G)}")
    return G


@app.cell
def _(G):
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    print("Some nodes in-degree vs out-degree:")
    for _node in list(G.nodes())[:10]:
        print(f"Node {_node}: in-degree={in_degrees[_node]}, out-degree={out_degrees[_node]}")

    avg_in = sum(in_degrees.values()) / len(in_degrees)
    avg_out = sum(out_degrees.values()) / len(out_degrees)
    print(f"\nAverage in-degree: {avg_in:.2f}, Average out-degree: {avg_out:.2f}")
    return avg_in, avg_out, in_degrees, out_degrees


@app.cell
def _(mo):
    mo.md(
        """
        In-links and out-links degrees are quite varied on the web as it could be expected (and viewed from the above in/out degree results). For example, while a site like Facebook has many in-links (due to popularity) there, a more obscure collection of blogs might have many out-links, but not many in-links.

        Average in-degrees and out-degrees are the same which makes sense for a directed graph.
        """
    )
    return


@app.cell
def _(in_degrees, out_degrees, plt):
    degree_hist_fig = plt.figure(figsize=(8, 4))
    plt.hist(list(in_degrees.values()), bins=20, alpha=0.7, label="in-degree")
    plt.hist(list(out_degrees.values()), bins=20, alpha=0.7, label="out-degree")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.legend()
    plt.title("In-degree vs Out-degree distribution")
    degree_hist_fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        The nodes (websites) have a relatively wide range of degrees ("in" and "out" hyperlinks), but nonetheless, most nodes exhibit a much lower degree of connectivity.
        """
    )
    return


@app.cell
def _(G, nx):
    weak_cc = list(nx.weakly_connected_components(G))
    print(f"Number of weakly connected components: {len(weak_cc)}")

    largest_weak = max(weak_cc, key=len)
    print(f"Largest weakly connected component size: {len(largest_weak)}")
    return largest_weak, weak_cc


@app.cell
def _(mo):
    mo.md(
        """
        All of the sampled nodes make one big weakly connected component (graph, or rather subgraph)
        """
    )
    return


@app.cell
def _(G, nx):
    strong_cc = list(nx.strongly_connected_components(G))
    print(f"Number of strongly connected components: {len(strong_cc)}")

    largest_strong = max(strong_cc, key=len)
    print(f"Largest strongly connected component size: {len(largest_strong)}")
    return largest_strong, strong_cc


@app.cell
def _(mo):
    mo.md(
        """
        There are more SCCs than WCCs which is to be expected.
        """
    )
    return


@app.cell
def _(G, largest_strong, nx, random):
    H = G.subgraph(largest_strong).copy()

    nodes = list(H.nodes())
    _u, _v = random.sample(nodes, 2)

    shortest_path = nx.shortest_path(H, source=_u, target=_v)
    print(f"Shortest path from {_u} to {_v}: {shortest_path}")

    print(f"Number of edges in this path: {len(shortest_path)-1}")
    return H, shortest_path


@app.cell
def _(mo):
    mo.md(
        """
        The above code shows the shortest path between 2 random nodes.
        """
    )
    return


@app.cell
def _(G, nx, plt, random):
    center_node = random.choice(list(G.nodes()))

    neighbors = set(G.successors(center_node))
    neighbors.update(G.predecessors(center_node))

    neighbors.add(center_node)

    subG = G.subgraph(neighbors).copy()

    neighborhood_fig = plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(subG)
    nx.draw(subG, pos, with_labels=True, node_size=500, node_color="skyblue", arrowsize=20)

    plt.title(f"Neighborhood of node {center_node}")
    neighborhood_fig
    return center_node, neighbors, subG


@app.cell
def _(mo):
    mo.md(
        """
        The above directed graph shows just a small sample of an interconnected network that is, even now, quite difficult to decipher due to the vast number of vertices and edges.
        """
    )
    return


if __name__ == "__main__":
    app.run()
