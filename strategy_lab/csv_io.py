#!/usr/bin/env python
"""CSV IO helpers for real-world A-share data files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def read_csv_robust(path: str | Path, encodings: Iterable[str] = DEFAULT_ENCODINGS, **kwargs) -> pd.DataFrame:
    """Read CSV with common China-market encodings."""
    errors: list[str] = []
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    joined = " | ".join(errors)
    raise UnicodeDecodeError("csv", b"", 0, 1, f"failed to decode {path}: {joined}")


def coerce_bool_series(series: pd.Series, default: bool | None = None) -> pd.Series:
    """Coerce common bool encodings without treating non-empty strings as True."""
    true_values = {"true", "t", "1", "yes", "y", "on", "是", "可交易", "交易", "正常"}
    false_values = {"false", "f", "0", "no", "n", "off", "否", "不可交易", "停牌", "暂停", "st"}

    def convert(value):
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return default

    return series.map(convert).astype("boolean")
