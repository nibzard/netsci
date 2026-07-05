"""Synthetic test fixtures.

These fixtures are SYNTHETIC data generated for unit testing only -- they do
not represent real Croatian tourism statistics. They exist to validate that
the pipeline code is correct; actual analysis results must come from the
real DZS export (see data/raw/DOWNLOAD_INSTRUCTIONS.md).
"""
import numpy as np
import pandas as pd
import pytest

from src.config import MONTHS

RNG = np.random.default_rng(0)

REGIONS = [
    ("Dubrovnik", "Dubrovacko-neretvanska", True),
    ("Split", "Splitsko-dalmatinska", True),
    ("Zadar", "Zadarska", True),
    ("Rovinj", "Istarska", True),
    ("Opatija", "Primorsko-goranska", True),
    ("Zagreb", "Grad Zagreb", False),
    ("Karlovac", "Karlovacka", False),
    ("Varazdin", "Varazdinska", False),
    ("Osijek", "Osjecko-baranjska", False),
    ("Plitvicka Jezera", "Licko-senjska", True),
]


def _seasonal_curve(coastal: bool) -> np.ndarray:
    if coastal:
        base = np.array([2, 2, 3, 5, 10, 18, 25, 23, 12, 6, 3, 2], dtype=float)
    else:
        base = np.array([7, 7, 8, 8, 9, 9, 10, 9, 9, 8, 8, 8], dtype=float)
    noise = RNG.normal(0, 0.5, size=12)
    curve = np.clip(base + noise, 0.1, None)
    return curve


@pytest.fixture
def synthetic_long_df() -> pd.DataFrame:
    rows = []
    for region, county, coastal in REGIONS:
        for year in (2022, 2023):
            curve = _seasonal_curve(coastal)
            total = RNG.integers(5_000, 200_000)
            nights = curve / curve.sum() * total
            for month, value in zip(MONTHS, nights):
                rows.append(
                    {
                        "region": region,
                        "county": county,
                        "year": year,
                        "month": month,
                        "indicator": "Nights",
                        "value": value,
                    }
                )
                rows.append(
                    {
                        "region": region,
                        "county": county,
                        "year": year,
                        "month": month,
                        "indicator": "Arrivals",
                        "value": value / 3,
                    }
                )
    return pd.DataFrame(rows)
