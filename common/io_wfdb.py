"""Read CTU-UHB WFDB signals and clinical fields from original headers."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import wfdb


@dataclass
class CTGRecord:
    signal: np.ndarray
    fs: float
    channel_names: list[str]
    units: list[str]
    clinical: dict[str, float | int | str]


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")


def parse_header_comments(header: str) -> dict[str, float | int | str]:
    """Parse '#field  value' while preserving original field names."""
    result: dict[str, float | int | str] = {}
    for line in header.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        content = line[1:].strip().split("\t:", 1)[0].strip()
        if not content or content.startswith("-"):
            continue
        parts = re.split(r"\s{2,}", content, maxsplit=1)
        if len(parts) != 2:
            parts = content.rsplit(None, 1)
        if len(parts) != 2:
            continue
        key, value = (part.strip() for part in parts)
        if not key or not value:
            continue
        if _NUMBER.fullmatch(value):
            number = float(value)
            result[key] = int(number) if number.is_integer() else number
        else:
            result[key] = value
    return result


def read_record(directory: Path, record_name: str) -> CTGRecord:
    """Read one local record. No signal or clinical values are printed."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", record_name):
        raise ValueError("Invalid record name")
    header = (directory / f"{record_name}.hea").read_text(encoding="utf-8-sig")
    record = wfdb.rdrecord(str(directory / record_name), physical=True)
    return CTGRecord(
        signal=record.p_signal,
        fs=float(record.fs),
        channel_names=list(record.sig_name),
        units=list(record.units),
        clinical=parse_header_comments(header),
    )
