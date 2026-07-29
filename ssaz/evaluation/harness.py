"""Benchmark harness that issues queries through the exact retriever code
used by :class:`~ssaz.engine.AzSearchEngine`, guaranteeing that benchmark
numbers reflect end-user behavior.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from ssaz.engine import AzSearchEngine
from ssaz.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass
class EvalQuery:
    """A benchmark query with gold relevance judgments.

    ``relevant_ids`` may contain chunk ids or doc ids; matching is by
    chunk id first, falling back to the chunk's ``doc_id`` so judgments
    at document granularity remain usable after re-chunking.
    """

    query: str
    relevant_ids: Set[str]
    domain: Optional[str] = None
    query_id: Optional[str] = None


@dataclass
class EvaluationReport:
    k: int
    n_queries: int
    aggregate: Dict[str, float]
    per_domain: Dict[str, Dict[str, float]]
    latency_ms_p50: float
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k, "n_queries": self.n_queries,
            "aggregate": self.aggregate, "per_domain": self.per_domain,
            "latency_ms_p50": self.latency_ms_p50, "config": self.config,
        }

    def to_markdown(self) -> str:
        embedder = self.config.get("embedder", "?")
        lines = [
            f"### Retrieval evaluation — {embedder} (k={self.k}, "
            f"{self.n_queries} queries, p50 {self.latency_ms_p50:.1f} ms)",
            "",
            "| Slice | MRR | Recall@%d | nDCG@%d | n |" % (self.k, self.k),
            "|---|---|---|---|---|",
        ]
        rows = [("all", self.aggregate)]
        rows += sorted(self.per_domain.items())
        for name, scores in rows:
            lines.append(
                "| %s | %.4f | %.4f | %.4f | %d |" % (
                    name, scores["mrr"], scores["recall"],
                    scores["ndcg"], int(scores["n"])),
            )
        return "\n".join(lines)


class EvaluationHarness:
    def __init__(self, engine: AzSearchEngine) -> None:
        self.engine = engine

    def evaluate(self, queries: Sequence[EvalQuery],
                 k: int = 5) -> EvaluationReport:
        per_query: List[Dict[str, Any]] = []
        latencies: List[float] = []
        for eval_query in queries:
            started = time.perf_counter()
            results = self.engine.search(eval_query.query, k=k)
            latencies.append((time.perf_counter() - started) * 1000.0)
            # Match on chunk id, falling back to doc id. Dedupe while
            # preserving rank order: several chunks of one document must
            # count as a single hit, not inflate recall/nDCG.
            retrieved = []
            seen = set()
            for result in results:
                hit_id = (result.chunk_id
                          if result.chunk_id in eval_query.relevant_ids
                          else result.doc_id)
                if hit_id not in seen:
                    seen.add(hit_id)
                    retrieved.append(hit_id)
            per_query.append({
                "domain": eval_query.domain or "unknown",
                "rr": reciprocal_rank(retrieved, eval_query.relevant_ids),
                "recall": recall_at_k(retrieved, eval_query.relevant_ids, k),
                "ndcg": ndcg_at_k(retrieved, eval_query.relevant_ids, k),
            })

        def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
            n = len(rows)
            if n == 0:
                return {"mrr": 0.0, "recall": 0.0, "ndcg": 0.0, "n": 0}
            return {
                "mrr": sum(r["rr"] for r in rows) / n,
                "recall": sum(r["recall"] for r in rows) / n,
                "ndcg": sum(r["ndcg"] for r in rows) / n,
                "n": n,
            }

        domains = sorted({row["domain"] for row in per_query})
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        return EvaluationReport(
            k=k, n_queries=len(per_query),
            aggregate=summarize(per_query),
            per_domain={d: summarize([r for r in per_query
                                      if r["domain"] == d])
                        for d in domains},
            latency_ms_p50=p50,
            config={"embedder": self.engine.embedder.name,
                    "index": type(self.engine.index).__name__},
        )

    @staticmethod
    def load_queries(path: Union[str, Path]) -> List[EvalQuery]:
        """Load benchmark queries; see :func:`ssaz.data.load_queries`"""
        from ssaz.data import load_queries
        return load_queries(path)
