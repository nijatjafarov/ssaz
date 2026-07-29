"""
All metrics take a ranked list of retrieved ids (best first) and the set
of relevant ids, with binary relevance.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Set


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mrr(all_retrieved: Iterable[List[str]],
        all_relevant: Iterable[Set[str]]) -> float:
    """Mean reciprocal rank over a query set"""
    ranks = [reciprocal_rank(retrieved, relevant)
             for retrieved, relevant in zip(all_retrieved, all_relevant)]
    return sum(ranks) / len(ranks) if ranks else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant items present in the top-k"""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Normalized discounted cumulative gain at k (binary relevance)"""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, item in enumerate(retrieved[:k], start=1)
              if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1)
               for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
