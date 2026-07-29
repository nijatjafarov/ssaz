"""
Default data loaders: files in, pipeline-ready objects out.


Custom field names: pass ``field_map`` to map your keys onto the
  canonical ones

      docs = load_documents("corpus.json",
                            field_map={"doc_id": "article_no",
                                       "text": "body",
                                       "date": "issued_on"})

      queries = load_queries("golden.json",
                             field_map={"query": "sual",
                                        "relevant_ids": "gold_docs"})

Custom file formats: register a reader for a new suffix:

      from ssaz import register_corpus_format

      @register_corpus_format(".csv")
      def read_csv(path):
          import csv
          with open(path, encoding="utf-8", newline="") as f:
              return list(csv.DictReader(f))   # -> List[dict]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from ssaz.documents import Document
from ssaz.evaluation.harness import EvalQuery
from ssaz.registry import Registry

_SCALARS = (str, int, float, bool)

# Canonical field to built-in alias chain
_DOC_ALIASES: Dict[str, Sequence[str]] = {
    "doc_id": ("doc_id", "id"),
    "text": ("text", "content", "body"),
    "domain": ("domain",),
    "title": ("title",),
    "date": ("date", "published_at"),
}
_QUERY_ALIASES: Dict[str, Sequence[str]] = {
    "query": ("query", "question"),
    "relevant_ids": ("relevant_ids", "source_doc_ids"),
    "answerable": ("answerable",),
    "domain": ("domain",),
    "query_id": ("query_id", "id"),
}

# File format registry
_FORMATS = Registry("corpus format")


def register_corpus_format(suffix: str,
                           reader: Optional[Callable[[Path], List[dict]]] = None):
    """Register a file reader for ``suffix``. The reader
    takes a :class:`Path` and returns a list of record dicts. Usable as a
    decorator"""
    return _FORMATS.register(suffix.lower(), reader)


def available_formats() -> list:
    return _FORMATS.names()


@register_corpus_format(".jsonl")
def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@register_corpus_format(".json")
def _read_json(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def _read_records(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix not in _FORMATS:
        raise ValueError(
            f"No reader registered for {suffix!r} files. Available: "
            f"{available_formats()}; add one with register_corpus_format().")
    return _FORMATS.create(suffix, path=path)


# Field resolution
def _resolve_keys(aliases: Dict[str, Sequence[str]],
                  field_map: Optional[Dict[str, Union[str, Sequence[str]]]],
                  ) -> Dict[str, Sequence[str]]:
    """Merge a user field_map into the alias table"""
    if not field_map:
        return aliases
    merged = dict(aliases)
    for canonical, user_keys in field_map.items():
        if canonical not in aliases:
            raise KeyError(
                f"Unknown canonical field {canonical!r}. "
                f"Options: {sorted(aliases)}")
        if isinstance(user_keys, str):
            user_keys = (user_keys,)
        merged[canonical] = tuple(user_keys) + tuple(aliases[canonical])
    return merged


def _get(record: dict, keys: Sequence[str], default=None):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


# Documents
def load_documents(path: Union[str, Path],
                   default_domain: str = "general",
                   field_map: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
                   ) -> List[Document]:
    """Load a corpus into pipeline-ready :class:`Document` objects.

    Args:
        path: Corpus file or directory.
        default_domain: Domain for records that specify none.
        field_map: Optional canonical-field -> your-key(s) mapping, e.g.
            ``{"doc_id": "article_no", "text": "body"}``.
    """
    path = Path(path)
    if path.is_dir():
        return [Document(doc_id=file.stem,
                         text=file.read_text(encoding="utf-8"),
                         domain=default_domain)
                for file in sorted(path.glob("**/*.txt"))]
    if path.suffix == ".txt":
        return [Document(doc_id=path.stem,
                         text=path.read_text(encoding="utf-8"),
                         domain=default_domain)]

    keys = _resolve_keys(_DOC_ALIASES, field_map)
    consumed = {alias for chain in keys.values() for alias in chain}
    consumed.add("metadata")
    documents = []
    for record in _read_records(path):
        doc_id = _get(record, keys["doc_id"])
        if not doc_id:
            raise ValueError(
                f"Document record has no {keys['doc_id']} key: {record!r:.80}")
        extras = {key: value for key, value in record.items()
                  if key not in consumed and isinstance(value, _SCALARS)}
        extras.update(record.get("metadata") or {})
        documents.append(Document(
            doc_id=str(doc_id),
            text=_get(record, keys["text"], ""),
            domain=_get(record, keys["domain"], default_domain),
            title=_get(record, keys["title"]),
            date=_get(record, keys["date"]),
            metadata=extras,
        ))
    return documents


# Queries
def load_queries(path: Union[str, Path],
                 documents: Optional[Iterable[Document]] = None,
                 field_map: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
                 ) -> List[EvalQuery]:
    """Load benchmark queries into :class:`EvalQuery` objects.

    Args:
        path: Simple SSAZ format, the default format, or your own layout 
            via ``field_map``. Unanswerable questions
            (``answerable: false``) carry no retrieval gold and are
            dropped.
        documents: Optional corpus the queries will be evaluated against.
            When given, each query's domain is derived from its gold
            document and queries whose gold documents are absent from the
            corpus are dropped.
        field_map: Optional canonical-field to your key mapping.
    """
    keys = _resolve_keys(_QUERY_ALIASES, field_map)
    doc_domains = ({d.doc_id: d.domain for d in documents}
                   if documents is not None else None)
    queries: List[EvalQuery] = []
    for record in _read_records(Path(path)):
        query = _get(record, keys["query"])
        relevant = set(_get(record, keys["relevant_ids"], []) or [])
        if not query or not relevant \
                or not _get(record, keys["answerable"], True):
            continue
        domain = _get(record, keys["domain"])
        if doc_domains is not None:
            relevant = {r for r in relevant if r in doc_domains}
            if not relevant:
                continue
            domain = domain or doc_domains[next(iter(relevant))]
        queries.append(EvalQuery(
            query=query,
            relevant_ids=relevant,
            domain=domain,
            query_id=_get(record, keys["query_id"]),
        ))
    return queries
