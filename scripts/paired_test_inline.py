"""
Inline paired-bootstrap test: re-run a small set of formats and compute
paired-bootstrap p-values directly between baselines and RODS variants.

Output: a markdown-friendly table.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.corpus import load_corpus
from benchmark.retrieval import RetrieverType, get_retriever
from benchmark.metrics import recall_at_k, mrr_at_k, paired_bootstrap_p
from benchmark.enterprise_eval import load_queries, get_format_modules

QUERIES_FILES = [
    ("original", Path(__file__).parent.parent / "data" / "enterprise_queries.json"),
    ("heldout",  Path(__file__).parent.parent / "data" / "enterprise_queries_heldout.json"),
]

FORMATS_TO_TEST = [
    "v0_baseline",      # L1
    "bl_semantic_chunks",  # L2
    "bl_summary_chunks",   # L3
    "dt5_query",        # docT5query baseline
    "rods_m4",
    "rods_m6",
    "v24_three_vector",
]

PAIRS = [
    ("v0_baseline", "rods_m6"),
    ("v0_baseline", "rods_m4"),
    ("v0_baseline", "dt5_query"),
    ("dt5_query", "rods_m6"),
    ("bl_semantic_chunks", "rods_m6"),
    ("v0_baseline", "v24_three_vector"),
]


def evaluate(format_name, fn, documents, queries, retriever_type):
    chunks = fn(documents)
    retriever = get_retriever(retriever_type)
    index = retriever.index(chunks)
    per_q = []
    for q in queries:
        targets = set(q["target_sections"])
        results = index.search(q["question"], top_k=min(20, len(chunks)))
        seen = []
        seen_set = set()
        for chunk, _ in results:
            if chunk.source_section not in seen_set:
                seen.append(chunk.source_section)
                seen_set.add(chunk.source_section)
            if len(seen) >= 20:
                break
        recall_5 = len(set(seen[:5]) & targets) / len(targets) if targets else 0.0
        recall_10 = len(set(seen[:10]) & targets) / len(targets) if targets else 0.0
        mrr = mrr_at_k(seen, targets, 10)
        per_q.append({"recall@5": recall_5, "recall@10": recall_10, "mrr@10": mrr})
    return per_q


def main(retriever_type="hybrid"):
    documents = load_corpus()
    fmt_mods = dict(get_format_modules(FORMATS_TO_TEST))
    if retriever_type in ("dense", "hybrid"):
        r = get_retriever(retriever_type)
        if hasattr(r, "_dense"):
            r._dense._model.encode(["warmup"], show_progress_bar=False)

    for label, path in QUERIES_FILES:
        with open(path) as f:
            queries = json.load(f)["queries"]
        print(f"\n=== {label.upper()}: {path.name} ({len(queries)} queries) ===")

        results_per_format = {}
        for name, fn in fmt_mods.items():
            results_per_format[name] = evaluate(name, fn, documents, queries, retriever_type)

        # Aggregate
        for name in FORMATS_TO_TEST:
            if name not in results_per_format: continue
            r5 = [x["recall@5"] for x in results_per_format[name]]
            r10 = [x["recall@10"] for x in results_per_format[name]]
            mrr = [x["mrr@10"] for x in results_per_format[name]]
            print(f"  {name:<22} R@5={sum(r5)/len(r5):.3f}  R@10={sum(r10)/len(r10):.3f}  MRR={sum(mrr)/len(mrr):.3f}")

        # Paired bootstrap
        print(f"\n  Paired-bootstrap p-values (R@5):")
        for a, b in PAIRS:
            if a not in results_per_format or b not in results_per_format: continue
            ra = [x["recall@5"] for x in results_per_format[a]]
            rb = [x["recall@5"] for x in results_per_format[b]]
            diff, p = paired_bootstrap_p(rb, ra, n_boot=2000)
            sig = "**" if p < 0.05 else "  "
            print(f"    {sig} {b:<22} - {a:<22}  Δ={diff:+.4f}  p={p:.3f}")

        print(f"\n  Paired-bootstrap p-values (MRR@10):")
        for a, b in PAIRS:
            if a not in results_per_format or b not in results_per_format: continue
            ra = [x["mrr@10"] for x in results_per_format[a]]
            rb = [x["mrr@10"] for x in results_per_format[b]]
            diff, p = paired_bootstrap_p(rb, ra, n_boot=2000)
            sig = "**" if p < 0.05 else "  "
            print(f"    {sig} {b:<22} - {a:<22}  Δ={diff:+.4f}  p={p:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dense", action="store_true")
    p.add_argument("--hybrid", action="store_true")
    args = p.parse_args()
    rt = "hybrid" if args.hybrid else ("dense" if args.dense else "tfidf")
    main(retriever_type=rt)
