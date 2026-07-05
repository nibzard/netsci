import numpy as np

from src.config import MONTHS
from src.preprocessing import build_node_table, build_seasonality_fingerprints


def test_seasonality_fingerprints_sum_to_one(synthetic_long_df):
    fp = build_seasonality_fingerprints(synthetic_long_df)
    sums = fp[MONTHS].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_seasonality_fingerprints_shape(synthetic_long_df):
    fp = build_seasonality_fingerprints(synthetic_long_df)
    n_regions = synthetic_long_df["region"].nunique()
    n_years = synthetic_long_df["year"].nunique()
    assert len(fp) == n_regions * n_years


def test_node_table_has_coastal_flag(synthetic_long_df):
    nodes = build_node_table(synthetic_long_df)
    assert "coastal" in nodes.columns
    assert nodes["coastal"].any()
    assert (~nodes["coastal"]).any()
