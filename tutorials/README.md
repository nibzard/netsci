# Tutorials

This directory contains all tutorial materials for the Network Analysis course, presented as Jupyter notebooks.

## Structure

Tutorials are organized to complement the lecture materials, providing hands-on experience with network analysis concepts and tools.

## Running the Notebooks

### Requirements

Use the project-local `.venv` managed by `uv`.

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the project environment in the project root
cd ..
uv sync
```

Do not rely on a checked-in `env/` directory. Recreate `.venv/` locally with `uv sync` when needed.

### Launch Jupyter

From the repository root:

```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

## Topics Covered

The tutorials cover practical implementations of the concepts discussed in lectures, including:

- Basic network creation and manipulation with NetworkX
- Computing network metrics and centrality measures
- Visualizing networks with different layouts and tools
- Implementing random graph models
- Community detection algorithms
- Network resilience analysis
- Temporal network analysis
- Real-world network data analysis
