"""Evaluation harness: MRR, Recall@k, nDCG@k with per-domain breakdown"""

from ssaz.evaluation.harness import EvalQuery, EvaluationHarness, EvaluationReport
from ssaz.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

__all__ = [
    "EvalQuery",
    "EvaluationHarness",
    "EvaluationReport",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
