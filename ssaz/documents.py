"""Document container used throughout the SSAZ pipeline"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """A source document to be chunked, enriched, embedded, and indexed.

    Attributes:
        doc_id: Stable identifier for the document.
        text: Full document text.
        domain: One of ``legal``, ``news``, ``encyclopedic`` or 
            any user-defined domain label. Drives
            structure-aware chunking defaults.
        title: Optional document title.
        date: Optional document date. If absent, the metadata
            enricher attempts to extract one from the text.
        metadata: Arbitrary extra metadata carried onto every chunk.
    """

    doc_id: str
    text: str
    domain: str = "general"
    title: Optional[str] = None
    date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("Document.doc_id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("Document.text must be a string")
