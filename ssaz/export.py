"""
Export mechanism: chunks, search results, queries, and evaluation
reports out to JSON, JSONL, CSV, or Markdown with one call.

To extend:

    from ssaz import register_export_format

    @register_export_format(".xml")
    def write_xml(records, path):
        ...
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from ssaz.registry import Registry

_EXPORTERS = Registry("export format")


def register_export_format(suffix: str,
                           writer: Optional[Callable[[List[dict], Path], None]] = None):
    return _EXPORTERS.register(suffix.lower(), writer)


def available_export_formats() -> list:
    return _EXPORTERS.names()


# Record normalization
def _normalize(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def to_records(items: Any) -> List[Dict[str, Any]]:
    """Normalize pipeline objects to a list of plain dicts."""
    if hasattr(items, "to_dict"):
        items = [items]
    elif isinstance(items, dict) or not isinstance(items, Iterable):
        items = [items]
    records: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, "to_dict"):
            record = item.to_dict()
        elif is_dataclass(item):
            record = asdict(item)
        elif isinstance(item, dict):
            record = dict(item)
        else:
            raise TypeError(
                f"Cannot export {type(item).__name__}; expected a "
                "dataclass, a dict, or an object with to_dict().")
        records.append({key: _normalize(value)
                        for key, value in record.items()})
    return records


# Built-in writers
@register_export_format(".json")
def _write_json(records: List[dict], path: Path) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                    encoding="utf-8")


@register_export_format(".jsonl")
def _write_jsonl(records: List[dict], path: Path) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8")


@register_export_format(".csv")
def _write_csv(records: List[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: (json.dumps(value, ensure_ascii=False)
                      if isinstance(value, (dict, list)) else value)
                for key, value in record.items()
            })


@register_export_format(".md")
def _write_md(records: List[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        text = str(value).replace("|", "\\|").replace("\n", " ")
        return text if len(text) <= 160 else text[:157] + "…"

    lines = ["| " + " | ".join(fields) + " |",
             "|" + "---|" * len(fields)]
    for record in records:
        lines.append("| " + " | ".join(cell(record.get(f, "")) for f in fields)
                     + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Entry point

def export_data(items: Any, path: Union[str, Path],
                format: Optional[str] = None) -> Path:
    """Export pipeline objects to ``path``.

    Args:
        items: Chunks, search results, eval queries, an evaluation
            report, or plain dicts (a single object or an iterable).
        path: Output file; the suffix selects the format unless
            ``format`` overrides it.
        format: Optional explicit format name.

    Returns the written path.
    """
    path = Path(path)
    suffix = ("." + format.lstrip(".").lower()) if format else path.suffix.lower()
    if suffix not in _EXPORTERS:
        raise ValueError(
            f"No exporter for {suffix!r}. Available: "
            f"{available_export_formats()}; add one with "
            "register_export_format().")
    records = to_records(items)
    _EXPORTERS.create(suffix, records=records, path=path)
    return path
