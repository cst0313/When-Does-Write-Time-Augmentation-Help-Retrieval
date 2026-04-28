"""Retrieval quality metrics."""

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class FormatResult:
    format_name: str
    format_desc: str
    # Per-query results: list of (rank | None)
    ranks: list[int | None] = field(default_factory=list)
    # Token counts per chunk
    chunk_token_counts: list[int] = field(default_factory=list)
    # Baseline token counts (V0) for efficiency ratio
    baseline_token_counts: list[int] | None = None

    def precision_at_k(self, k: int) -> float:
        """Fraction of queries where ground truth chunk is in top-k."""
        found = sum(1 for r in self.ranks if r is not None and r <= k)
        return found / len(self.ranks) if self.ranks else 0.0

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank."""
        reciprocals = [1.0 / r if r is not None else 0.0 for r in self.ranks]
        return mean(reciprocals) if reciprocals else 0.0

    @property
    def avg_tokens_per_chunk(self) -> float:
        return mean(self.chunk_token_counts) if self.chunk_token_counts else 0.0

    @property
    def token_efficiency(self) -> float:
        """Ratio of baseline tokens to this format's tokens (>1 means more tokens than baseline)."""
        if not self.chunk_token_counts or not self.baseline_token_counts:
            return 1.0
        return mean(self.baseline_token_counts) / mean(self.chunk_token_counts)

    def summary(self) -> dict:
        return {
            "format": self.format_name,
            "description": self.format_desc,
            "p@1": round(self.precision_at_k(1), 3),
            "p@3": round(self.precision_at_k(3), 3),
            "mrr": round(self.mrr, 3),
            "avg_tokens": round(self.avg_tokens_per_chunk, 1),
            "token_efficiency": round(self.token_efficiency, 3),
            "n_queries": len(self.ranks),
        }


def count_tokens_approx(text: str) -> int:
    """Approximate token count: ~4 chars per token (rough but fast)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# IR-standard metrics for BEIR-style evaluation.
# Each query has a set of relevant documents (qrels). Retriever returns a
# ranked list. We compute Recall@K, MRR, nDCG@K.
# ---------------------------------------------------------------------------
import math


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k."""
    if not relevant:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return len(top_k & relevant) / len(relevant)


def mrr_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int = 10) -> float:
    """Reciprocal of the rank of the first relevant doc; 0 if none in top-k."""
    for i, did in enumerate(ranked_doc_ids[:k], start=1):
        if did in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """nDCG@K with binary or graded relevance. qrels maps doc_id -> relevance score."""
    # DCG of returned ranking
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k], start=1):
        rel = qrels.get(did, 0)
        if rel > 0:
            # Use 2^rel - 1 form (standard for graded relevance)
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    # Ideal DCG: rank by relevance descending
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, start=1) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics(per_query_metrics: list[dict]) -> dict[str, float]:
    """Mean of each metric across queries."""
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {k: sum(m[k] for m in per_query_metrics) / len(per_query_metrics) for k in keys}


def bootstrap_ci(per_query_values: list[float], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """
    Bootstrap 95%-CI of the mean of a per-query metric.
    Returns (mean, lower, upper).
    """
    import random
    if not per_query_values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(per_query_values)
    means = []
    for _ in range(n_boot):
        sample = [per_query_values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    mean = sum(per_query_values) / n
    return mean, lo, hi


def paired_bootstrap_p(per_query_a: list[float], per_query_b: list[float],
                        n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """
    Paired-bootstrap p-value for H0: mean(A) = mean(B).
    Returns (mean_diff_AminusB, p_value).
    """
    import random
    assert len(per_query_a) == len(per_query_b), "must be paired per-query"
    n = len(per_query_a)
    if n == 0:
        return 0.0, 1.0
    diffs = [a - b for a, b in zip(per_query_a, per_query_b)]
    observed = sum(diffs) / n
    rng = random.Random(seed)
    centered = [d - observed for d in diffs]
    extreme = 0
    for _ in range(n_boot):
        sample = [centered[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        if abs(m) >= abs(observed):
            extreme += 1
    return observed, extreme / n_boot
