"""Acquisition of the DZS (Croatian Bureau of Statistics) tourism dataset.

Two paths are supported:

1. ``fetch_from_pxweb_api`` — calls the live PxWeb REST API for Table 1.6
   ("Accommodation capacities, tourist arrivals and nights ... by towns and
   municipalities, by months", code ``BS_TU16.px``) and writes a CSV to
   ``data/raw/``. Requires network access to ``web.dzs.hr``.

2. ``load_manual_export`` — reads a CSV/PX file that was downloaded by hand
   through the PxWeb web UI (see ``data/raw/DOWNLOAD_INSTRUCTIONS.md``).

The query body in ``PXWEB_QUERY`` reflects the table's structure as of the
time this module was written. PxWeb table/variable codes occasionally change
between DZS data releases; if the API call fails, open the interactive
selector (URL in DOWNLOAD_INSTRUCTIONS.md), make a selection, and use the
"API" link it generates to obtain an up-to-date query body.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_MUNICIPALITY_FILE

logger = logging.getLogger(__name__)

PXWEB_BASE = "https://web.dzs.hr/PXWeb/api/v1/en"
PXWEB_TABLE_PATH = f"{PXWEB_BASE}/Turizam/BS_TU16.px"

# JSON-stat2 query selecting every region, every available time period, and
# the Arrivals / Nights / Permanent beds variables. The exact variable codes
# ("Mjesto", "Mjesec", "ContentsCode") must be confirmed against the live
# metadata response (GET PXWEB_TABLE_PATH) before running in an environment
# with network access, since PxWeb assigns codes per table.
PXWEB_QUERY = {
    "query": [
        {"code": "Mjesto", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "Mjesec", "selection": {"filter": "all", "values": ["*"]}},
        {
            "code": "ContentsCode",
            "selection": {"filter": "item", "values": ["Dolasci", "Nocenja", "Postelje"]},
        },
    ],
    "response": {"format": "csv"},
}


def fetch_table_metadata(timeout: int = 30) -> dict:
    """Fetch PxWeb table metadata (variable codes and valid values)."""
    resp = requests.get(PXWEB_TABLE_PATH, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_from_pxweb_api(out_path: Path = RAW_MUNICIPALITY_FILE, timeout: int = 60) -> Path:
    """POST the JSON-stat2 query and save the response as CSV.

    Raises requests.RequestException on network failure -- callers should
    fall back to the manual-export instructions in that case.
    """
    resp = requests.post(PXWEB_TABLE_PATH, data=json.dumps(PXWEB_QUERY), timeout=timeout)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    logger.info("Saved PxWeb export to %s (%d bytes)", out_path, len(resp.content))
    return out_path


def load_manual_export(path: Path = RAW_MUNICIPALITY_FILE) -> pd.DataFrame:
    """Load a CSV that was exported manually from the PxWeb web UI.

    Delegates to ``src.preprocessing.load_raw_municipality_data``, which
    handles PxWeb's actual export quirks (variable delimiter, encoding,
    leading title/blank rows, the wide year-month-indicator column layout,
    and the English-language county hierarchy) and returns the canonical
    long-format table (``region``, ``year``, ``month``, ``indicator``,
    ``value``, ``county``).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Follow data/raw/DOWNLOAD_INSTRUCTIONS.md to "
            "obtain the DZS Table 1.6 export before running preprocessing."
        )
    from src.preprocessing import load_raw_municipality_data

    return load_raw_municipality_data(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["api", "validate"], default="validate")
    parser.add_argument("--out", type=Path, default=RAW_MUNICIPALITY_FILE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.mode == "api":
        fetch_from_pxweb_api(args.out)
    else:
        df = load_manual_export(args.out)
        logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], args.out)


if __name__ == "__main__":
    main()
