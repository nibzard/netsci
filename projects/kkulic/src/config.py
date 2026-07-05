"""Shared paths and constants for the Croatian tourism network project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

RAW_MUNICIPALITY_FILE = DATA_RAW / "dzs_tur16_municipality_monthly.csv"
RAW_ORIGIN_COUNTY_FILE = DATA_RAW / "dzs_origin_country_by_county.csv"

NODE_TABLE = DATA_PROCESSED / "nodes.parquet"
SEASONALITY_TABLE = DATA_PROCESSED / "seasonality_fingerprints.parquet"
EDGE_TABLE = DATA_PROCESSED / "edges.parquet"
GRAPH_FILE = DATA_PROCESSED / "tourism_network.graphml"
TEMPORAL_GRAPH_DIR = DATA_PROCESSED / "temporal_graphs"
TEMPORAL_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Coastal counties (along the Adriatic) vs. continental counties, used as a
# ground-truth label for community-detection validation and the ML
# classification task. Based on official Croatian county (zupanija) borders.
COASTAL_COUNTIES = {
    "Istarska", "Primorsko-goranska", "Licko-senjska",
    "Zadarska", "Sibensko-kninska", "Splitsko-dalmatinska",
    "Dubrovacko-neretvanska",
}

# All 21 first-level administrative units (20 counties + the City of
# Zagreb, which has county status) in the Republic of Croatia. Used to
# recognize the county-header rows that PxWeb inserts when a "counties,
# towns, municipalities" hierarchical selection is exported -- the raw
# file then lists each county's own aggregate row immediately followed by
# its constituent towns/municipalities, in that fixed order.
CROATIAN_COUNTIES = COASTAL_COUNTIES | {
    "Zagrebacka", "Krapinsko-zagorska", "Sisacko-moslavacka", "Karlovacka",
    "Varazdinska", "Koprivnicko-krizevacka", "Bjelovarsko-bilogorska",
    "Viroviticko-podravska", "Pozesko-slavonska", "Brodsko-posavska",
    "Osjecko-baranjska", "Vukovarsko-srijemska", "Medjimurska",
    "Grad Zagreb",
}

# Default similarity threshold / top-k for sparsifying the seasonality graph.
EDGE_SIMILARITY_THRESHOLD = 0.97
EDGE_TOP_K = 8

RANDOM_SEED = 42
