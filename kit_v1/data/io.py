"""OHLCV reading helpers shared by the data-preparation scripts and the dataset.

The training/eval code always works with the canonical lower-case columns
``datetime, open, high, low, close, volume`` (see ``normalize.fit_stats`` and
``infer.window``).  Raw vendor dumps are often capitalised and/or use ``date``
instead of ``datetime``, so everything goes through :func:`canon_ohlcv_columns`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union

import pandas as pd

# canonical column order used everywhere in kit_v1
CANONICAL_COLUMNS: tuple = ("datetime", "open", "high", "low", "close", "volume")

# aliases found in the wild -> canonical name
_ALIASES = {
    "date": "datetime",
    "time": "datetime",
    "timestamp": "datetime",
    "datetime64": "datetime",
    "vol": "volume",
    "turnover_volume": "volume",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}


def canon_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with canonical lower-case OHLCV column names.

    Unknown columns are dropped; the result keeps the canonical column order
    (all six columns are guaranteed to exist).  Raises ``ValueError`` if a
    required column cannot be found.
    """
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        renamed[col] = _ALIASES.get(key, key)
    out = df.rename(columns=renamed)
    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"OHLCV data is missing columns {missing}; got {list(df.columns)}")
    out = out[list(CANONICAL_COLUMNS)].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for c in CANONICAL_COLUMNS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def read_ohlcv(path: Union[str, Path],
               columns: Sequence[str] = CANONICAL_COLUMNS,
               end_ts=None) -> pd.DataFrame:
    """Read one parquet file and return it with canonical columns.

    If ``end_ts`` is given, only rows with ``datetime <= end_ts`` are kept
    (applied after canonicalisation so a ``date`` column also matches).
    """
    df = pd.read_parquet(path)
    df = canon_ohlcv_columns(df)
    if end_ts is not None:
        df = df[df["datetime"] <= pd.Timestamp(end_ts)]
    return df[list(columns)].reset_index(drop=True)


def list_parquet(directory: Union[str, Path]) -> list:
    """Sorted list of parquet files directly inside ``directory`` (empty if missing)."""
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(q for q in p.glob("*.parquet"))


def code_from_path(path: Union[str, Path]) -> str:
    """Instrument code = parquet file stem."""
    return Path(path).stem


__all__ = ["CANONICAL_COLUMNS", "canon_ohlcv_columns", "read_ohlcv",
           "list_parquet", "code_from_path"]
