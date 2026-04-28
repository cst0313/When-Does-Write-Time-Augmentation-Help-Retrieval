"""
Cross-encoder reranker test.

For each format, retrieve top-K=20 with hybrid retriever, then rerank
those 20 with a cross-encoder (ms-marco-MiniLM-L-6-v2). Compare RODS-M6
vs L1 baseline post-rerank. If RODS+rerank > L1+rerank, the schema adds
information beyond what the bi-encoder + reranker stack extracts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.corpus import load_corpus
from benchmark.retrieval import get_retriever
from benchmark.metrics import recall_at_k, mrr_at_k, paired_bootstrap_p
from benchmark.enterprise_eval import get_format_modules, CATEGORIES

DATA = Path(__file__).parent.parent / "data"

CROSS_ENCODER_PATH = "/tmp/cross_encoder"

FORMATS = ["v0_baseline", "rods_m4", "rods_m6", "v24_three_vector", "dt5_query"]


def evaluate_with_rerank(name, fn, queries, documents, reranker, top_first=20, top_final=10):
    chunks = fn(documents)
    retriever = get_retriever("hybrid")
    idx = retriever.index(chunks)

    per_q = []
    for q in queries:
        targets = set(q["target_sections"])
        # Stage 1: retrieve top_first hybrid candidates
        results = idx.search(q["question"], top_k=top_first)

        # Build (query, chunk_text) pairs and rerank
        pairs = [(q["question"], c.text[:512]) for c, _ in results]
        if pairs:
            scores = reranker.predict(pairs, show_progress_bar=False)
            # Pair with chunks and sort by reranker score
            paired = list(zip(results, scores))
            paired.sort(key=lambda x: -x[1])
            results = [r for r, _ in paired]

        # Dedupe by source_section, keep first
        seen, seen_set = [], set()
        for chunk, _ in results:
            if chunk.source_section not in seen_set:
                seen.append(chunk.source_section); seen_set.add(chunk.source_section)
            if len(seen) >= top_final: break

        r5 = len(set(seen[:5]) & targets) / len(targets) if targets else 0.0
        mrr = mrr_at_k(seen, targets, 10)
        per_q.append({"r5": r5, "mrr": mrr, "category": q["category"]})
    return per_q


def main():
    documents = load_corpus()
    fmt_mods = dict(get_format_modules(FORMATS))

    queries = []
    for path in [DATA / "enterprise_queries.json", DATA / "enterprise_queries_heldout.json"]:
        with open(path) as f:
            queries.extend(json.load(f)["queries"])
    print(f"Eval on {len(queries)} combined queries\n")

    print("Loading cross-encoder...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CROSS_ENCODER_PATH, max_length=512)
    print("Cross-encoder loaded.\n")

    # Warm dense
    r = get_retriever("hybrid")
    if hasattr(r, "_dense"):
        r._dense._model.encode(["warmup"], show_progress_bar=False)

    rows_per_format = {}
    print(f"{'Format':<22} {'R@5 (hybrid)':>14} {'R@5 (rerank)':>14} {'MRR@10 (rerank)':>16}")
    print("─" * 70)

    # First, compute baseline (no rerank) numbers from the standard eval
    for name in FORMATS:
        chunks = fmt_mods[name](documents)
        idx = get_retriever("hybrid").index(chunks)
        per_q_no_rr = []
        for q in queries:
            targets = set(q["target_sections"])
            results = idx.search(q["question"], top_k=10)
            seen, seen_set = [], set()
            for chunk, _ in results:
                if chunk.source_section not in seen_set:
                    seen.append(chunk.source_section); seen_set.add(chunk.source_section)
            r5 = len(set(seen[:5]) & targets) / len(targets) if targets else 0.0
            mrr = mrr_at_k(seen, targets, 10)
            per_q_no_rr.append({"r5": r5, "mrr": mrr})
        # With reranker
        per_q_rr = evaluate_with_rerank(name, fmt_mods[name], queries, documents, reranker)
        rows_per_format[name] = {"no_rr": per_q_no_rr, "rr": per_q_rr}
        r5_nr = sum(p["r5"] for p in per_q_no_rr) / len(per_q_no_rr)
        r5_rr = sum(p["r5"] for p in per_q_rr) / len(per_q_rr)
        mrr_rr = sum(p["mrr"] for p in per_q_rr) / len(per_q_rr)
        print(f"{name:<22} {r5_nr:>14.3f} {r5_rr:>14.3f} {mrr_rr:>16.3f}")

    # Significance: RODS-M6 vs L1 post-rerank
    l1_rr = [p["r5"] for p in rows_per_format["v0_baseline"]["rr"]]
    m6_rr = [p["r5"] for p in rows_per_format["rods_m6"]["rr"]]
    diff, p = paired_bootstrap_p(m6_rr, l1_rr, n_boot=2000)
    print(f"\nRODS-M6 vs L1 (post-rerank, R@5): Δ={diff:+.4f}  p={p:.3f}")
    l1_mrr_rr = [p["mrr"] for p in rows_per_format["v0_baseline"]["rr"]]
    m6_mrr_rr = [p["mrr"] for p in rows_per_format["rods_m6"]["rr"]]
    diff, p = paired_bootstrap_p(m6_mrr_rr, l1_mrr_rr, n_boot=2000)
    print(f"RODS-M6 vs L1 (post-rerank, MRR@10): Δ={diff:+.4f}  p={p:.3f}")

    # docT5query vs RODS post-rerank
    d_rr = [p["r5"] for p in rows_per_format["dt5_query"]["rr"]]
    diff, p = paired_bootstrap_p(m6_rr, d_rr, n_boot=2000)
    print(f"RODS-M6 vs docT5query (post-rerank, R@5): Δ={diff:+.4f}  p={p:.3f}")

    # Save raw for paper
    out_path = DATA / "reranker_results.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in rows_per_format.items()}, f)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
