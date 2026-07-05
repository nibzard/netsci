"""Clean the raw DZS export and derive the node table + seasonality fingerprints.

Expected raw schema (PxWeb CSV export, long format), one row per
(region, year, month, indicator):

    region;year;month;indicator;value

where ``region`` is the town/municipality name, ``indicator`` is one of
``Arrivals``, ``Nights``, ``Permanent beds``, and ``value`` is numeric.
PxWeb's actual column headers vary by export ("Towns/Municipalities",
"Mjesto", etc.); ``_normalize_columns`` maps common variants onto this
canonical schema. If a manually-downloaded file uses a wide layout
(months as columns) instead, ``_wide_to_long`` reshapes it first.

Output artifacts (written to data/processed/):

* ``nodes.parquet`` -- one row per municipality with county, coastal flag,
  and aggregate yearly totals (arrivals, nights, beds).
* ``seasonality_fingerprints.parquet`` -- one row per (municipality, year)
  with a 12-value normalized monthly-nights distribution. This is the
  feature vector used for edge construction.
"""
from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

from src.config import (
    COASTAL_COUNTIES,
    CROATIAN_COUNTIES,
    DATA_PROCESSED,
    MONTHS,
    NODE_TABLE,
    RAW_MUNICIPALITY_FILE,
    RAW_ORIGIN_COUNTY_FILE,
    SEASONALITY_TABLE,
)

logger = logging.getLogger(__name__)

_COLUMN_ALIASES = {
    "region": {"region", "towns/municipalities", "mjesto", "naselje", "town", "municipality", "spatial unit"},
    "year": {"year", "godina"},
    "month": {"month", "mjesec"},
    "indicator": {"indicator", "contentscode", "sadrzaj"},
    "value": {"value", "vrijednost"},
    "county": {"county", "zupanija", "county/region"},
}

# The real PxWeb manual export is wide: one row per region, with a
# repeated "{year} {month:02d} {indicator label}" column for every
# year-month (room counts, beds, arrivals/nights split by
# total/domestic/foreign). e.g. "2019 01 Tourist nights - total".
_PXWEB_WIDE_COL_RE = re.compile(r"^(\d{4})\s+(\d{2})\s+(.+)$")

_PXWEB_INDICATOR_LABELS = {
    "tourist arrivals - total": "Arrivals",
    "tourist nights - total": "Nights",
    "number of permanent beds": "Permanent beds",
}


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        for canonical, aliases in _COLUMN_ALIASES.items():
            if key in aliases:
                rename[col] = canonical
                break
    return df.rename(columns=rename)


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape a region x month wide table into the canonical long format."""
    month_cols = [c for c in df.columns if c in MONTHS]
    if not month_cols:
        return df
    id_cols = [c for c in df.columns if c not in month_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=month_cols, var_name="month", value_name="value")
    return long_df


# PxWeb's English-language hierarchical export uses "County of <name>" for
# the 20 counties and "City of Zagreb" for Croatia's one county-level city,
# e.g. "County of Primorje Gorski-kotar", "County of Istria", "City of
# Zagreb" -- not the Croatian zupanija names. It also prepends NUTS-level
# rollup rows above the counties: "Republic of Croatia", "Adriatic
# Croatia", "Continental Croatia".
_COUNTY_OF_RE = re.compile(r"^county of\s+(.+)$", re.IGNORECASE)
_CITY_OF_RE = re.compile(r"^city of\s+(.+)$", re.IGNORECASE)
_TOP_LEVEL_AGGREGATES = {"republic of croatia", "adriatic croatia", "continental croatia"}

# Substrings (diacritic-stripped, lowercased) that uniquely identify each
# of the 7 Adriatic counties regardless of whether the county name is
# given in Croatian ("Istarska") or PxWeb's English translation ("County
# of Istria").
_COASTAL_KEYWORDS = ("primorje", "istra", "istria", "lika", "zadar", "sibenik", "split", "dubrovnik")


def _is_county_row(region: str) -> bool:
    """True if ``region`` is a Croatian-language county-aggregate row,
    e.g. "Istarska" or "Istarska zupanija"/"Istarska županija".
    """
    key = _strip_diacritics(str(region)).strip().lower()
    key = re.sub(r"\s*zupanij[ae]\s*$", "", key)
    return key in {c.lower() for c in CROATIAN_COUNTIES}


def _is_coastal(county) -> bool:
    if pd.isna(county):
        return False
    if county in COASTAL_COUNTIES:
        return True
    key = _strip_diacritics(str(county)).lower()
    return any(kw in key for kw in _COASTAL_KEYWORDS)


def _assign_counties_from_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """PxWeb's hierarchical "counties, towns, municipalities" selection
    exports a flat list where each county's own aggregate row is
    immediately followed by its constituent towns/municipalities, in that
    fixed order (and may be wrapped in NUTS-level rollup rows above it).
    Detect the county-header rows -- either Croatian county names or
    English "County of <name>"/"City of Zagreb" -- and forward-fill the
    county onto the rows that follow. "County of <name>" headers are then
    dropped (their values are aggregates over the towns below them); "City
    of Zagreb" has no children and is kept as a node in its own right.
    """
    region_clean = df["region"].astype(str).str.strip()
    region_norm = region_clean.map(_strip_diacritics)

    is_hr_county = region_clean.map(_is_county_row)
    is_county_of = region_norm.str.match(_COUNTY_OF_RE)
    is_city_of = region_norm.str.match(_CITY_OF_RE)
    is_top_level = region_norm.str.lower().isin(_TOP_LEVEL_AGGREGATES)

    is_header = is_hr_county | is_county_of | is_city_of
    if not is_header.any():
        return df

    df = df.copy()
    df["county"] = region_clean.where(is_header).ffill()
    drop_mask = is_top_level | is_hr_county | is_county_of
    return df[~drop_mask].reset_index(drop=True)


def _melt_pxweb_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the real PxWeb manual-export layout (one row per region,
    "{year} {month:02d} {indicator label}" columns) into the canonical
    long format. Keeps only the *total* arrivals/nights and the permanent
    beds columns; domestic/foreign splits and room counts are dropped.
    """
    id_cols = [c for c in ("region", "county") if c in df.columns]
    value_cols = [c for c in df.columns if _PXWEB_WIDE_COL_RE.match(c)]
    long_df = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="raw_col", value_name="value")
    parsed = long_df["raw_col"].str.extract(_PXWEB_WIDE_COL_RE)
    parsed.columns = ["year", "month_num", "label"]
    long_df = pd.concat([long_df.drop(columns="raw_col"), parsed], axis=1)
    label_norm = long_df["label"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    long_df["indicator"] = label_norm.map(_PXWEB_INDICATOR_LABELS)
    long_df = long_df.dropna(subset=["indicator"]).copy()
    month_lookup = {f"{i:02d}": MONTHS[i - 1] for i in range(1, 13)}
    long_df["month"] = long_df["month_num"].map(month_lookup)
    long_df["year"] = long_df["year"].astype(int)
    return long_df[id_cols + ["year", "month", "indicator", "value"]]


def _detect_delimiter(line: str) -> str:
    """PxWeb exports quote every field (``"Spatial unit" "2019 01 ..."``)
    and separate them with ``;``, ``,``, a literal tab, or a plain space
    depending on the "Save result as..." option chosen; pick whichever
    appears most often in the header line (each is quoted/escaped
    consistently within a given export, so a raw count is a reliable
    signal even though field labels themselves may contain commas/spaces).
    """
    counts = {delim: line.count(delim) for delim in (";", "\t", ",", " ")}
    return max(counts, key=counts.get)


def _read_raw_csv(path) -> pd.DataFrame:
    """Read the raw export, trying several encodings, auto-detecting the
    field delimiter, and skipping any leading PxWeb title/blank rows
    before the real header row.

    The title row is a single quoted string (only 2 quote characters); the
    header/data rows are made of dozens of quoted fields and so contain
    many quote characters. That quote count -- not delimiter counts, since
    the delimiter itself varies (``;``/``,``/tab/space) -- is what reliably
    distinguishes a title/blank line from a real row.
    """
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin1"):
        try:
            with open(path, encoding=encoding) as f:
                lines = [f.readline() for _ in range(5)]
            skiprows = 0
            header_line = lines[0]
            for line in lines:
                if line.count('"') >= 10:
                    header_line = line
                    break
                skiprows += 1
            delimiter = _detect_delimiter(header_line)
            return pd.read_csv(
                path, sep=delimiter, engine="python", encoding=encoding, skiprows=skiprows, skipinitialspace=True
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def load_raw_municipality_data(path=RAW_MUNICIPALITY_FILE) -> pd.DataFrame:
    df = _read_raw_csv(path)
    df = _normalize_columns(df)

    if "region" in df.columns and "county" not in df.columns:
        df = _assign_counties_from_hierarchy(df)

    if any(_PXWEB_WIDE_COL_RE.match(c) for c in df.columns):
        df = _melt_pxweb_wide(df)
    else:
        df = _wide_to_long(df)

    required = {"region", "year", "month", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw file {path} is missing expected columns {missing} after "
            "normalization. Inspect the actual PxWeb export header and "
            "extend _COLUMN_ALIASES / _wide_to_long accordingly."
        )
    if "indicator" not in df.columns:
        df["indicator"] = "Nights"
    df["region"] = df["region"].astype(str).str.strip().map(_strip_diacritics)
    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df


def build_seasonality_fingerprints(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (region, year): 12 normalized monthly-nights shares."""
    nights = long_df[long_df["indicator"].str.contains("Night", case=False, na=False)]
    if nights.empty:
        nights = long_df
    pivot = nights.pivot_table(
        index=["region", "year"], columns="month", values="value", aggfunc="sum", fill_value=0.0
    )
    pivot = pivot.reindex(columns=MONTHS, fill_value=0.0)
    totals = pivot.sum(axis=1)
    fingerprints = pivot.div(totals.replace(0, pd.NA), axis=0).fillna(0.0)
    fingerprints["total_nights"] = totals
    fingerprints = fingerprints.reset_index()
    return fingerprints


def build_node_table(long_df: pd.DataFrame, origin_path=RAW_ORIGIN_COUNTY_FILE) -> pd.DataFrame:
    """Aggregate per-region totals and attach county / coastal-flag attributes."""
    agg = (
        long_df.groupby(["region", "indicator"])["value"]
        .sum()
        .unstack("indicator", fill_value=0.0)
        .reset_index()
    )
    agg.columns.name = None

    if "county" in long_df.columns:
        county_map = long_df.drop_duplicates("region").set_index("region")["county"]
        agg["county"] = agg["region"].map(county_map)
    else:
        agg["county"] = pd.NA
        logger.warning(
            "No county column found in raw data; coastal/continental flag "
            "will be NA. Supply a county mapping (e.g. via a lookup table) "
            "to populate it."
        )

    agg["coastal"] = agg["county"].map(_is_coastal)

    if origin_path.exists():
        origin = pd.read_csv(origin_path, sep=None, engine="python", decimal=",")
        origin = _normalize_columns(origin)
        if "county" in origin.columns:
            feeder_cols = [c for c in origin.columns if c not in {"county", "year"}]
            if feeder_cols:
                origin["dominant_feeder_market"] = origin[feeder_cols].idxmax(axis=1)
                agg = agg.merge(
                    origin[["county", "dominant_feeder_market"]].drop_duplicates("county"),
                    on="county",
                    how="left",
                )
    else:
        logger.info(
            "Optional origin-country file %s not found; skipping feeder-market "
            "attribute. See data/raw/DOWNLOAD_INSTRUCTIONS.md.",
            origin_path,
        )

    return agg


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    long_df = load_raw_municipality_data()
    nodes = build_node_table(long_df)
    fingerprints = build_seasonality_fingerprints(long_df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(NODE_TABLE, index=False)
    fingerprints.to_parquet(SEASONALITY_TABLE, index=False)
    logger.info("Wrote %d nodes to %s", len(nodes), NODE_TABLE)
    logger.info("Wrote %d (region, year) fingerprints to %s", len(fingerprints), SEASONALITY_TABLE)


if __name__ == "__main__":
    run()
