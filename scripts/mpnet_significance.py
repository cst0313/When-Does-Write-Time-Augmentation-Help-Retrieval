"""
Paired-bootstrap significance for mpnet on enterprise (orig + held-out).

Re-runs L1 / RODS-M4 / RODS-M6 / docT5query / three-vector under mpnet,
collects per-query R@5 and MRR@10, and reports paired bootstrap p-values
versus the L1 heading-aware baseline. Mirrors paired_test_inline.py
but for the larger embedder.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force mpnet
os.environ["AMD_SENTENCE_TRANSFORMER_PATH"] = "/tmp/mpnet_model"

from benchmark.corpus import load_corpus
from benchmark.retrieval import get_retriever
from benchmark.metrics import mrr_at_k, paired_bootstrap_p
from benchmark.enterprise_eval import get_format_modules

DATA = Path(__file__).parent.parent / "data"
FORMATS = ["v0_baseline", "rods_m4", "rods_m6", "dt5_query",
           "v24_three_vector", "v71_v64_synonym_chunk"]


def per_query_metrics(name, fn, queries, documents):
    chunks = fn(documents)
    idx = get_retriever("hybrid").index(chunks)
    rows = []
    for q in queries:
        targets = set(q["target_sections"])
        results = idx.search(q["question"], top_k=20)
        seen, seen_set = [], set()
        for chunk, _ in results:
            if chunk.source_section not in seen_set:
                seen.append(chunk.source_section)
                seen_set.add(chunk.source_section)
        r5 = len(set(seen[:5]) & targets) / len(targets) if targets else 0.0
        mrr = mrr_at_k(seen, targets, 10)
        rows.append({"r5": r5, "mrr": mrr})
    return rows


def main():
    documents = load_corpus()
    fmt_mods = dict(get_format_modules(FORMATS))

    r = get_retriever("hybrid")
    if hasattr(r, "_dense"):
        r._dense._model.encode(["warmup"], show_progress_bar=False)
        print(f"  embedder dim = {r._dense._model.get_embedding_dimension()}")

    out = {}
    for q_path, label in [(DATA / "enterprise_queries.json", "orig"),
                          (DATA / "enterprise_queries_heldout.json", "heldout")]:
        with open(q_path) as f:
            queries = json.load(f)["queries"]
        print(f"\n=== {label} (n={len(queries)}) ===")

        rows_per_format = {}
        for name in FORMATS:
            rows_per_format[name] = per_query_metrics(name, fmt_mods[name], queries, documents)

        l1_r5 = [r["r5"] for r in rows_per_format["v0_baseline"]]
        l1_mrr = [r["mrr"] for r in rows_per_format["v0_baseline"]]
        for name in FORMATS:
            r5 = [r["r5"] for r in rows_per_format[name]]
            mrr = [r["mrr"] for r in rows_per_format[name]]
            d_r5, p_r5 = paired_bootstrap_p(r5, l1_r5, n_boot=2000)
            d_mrr, p_mrr = paired_bootstrap_p(mrr, l1_mrr, n_boot=2000)
            mr5 = sum(r5) / len(r5)
            mmr = sum(mrr) / len(mrr)
            print(f"  {name:<22}  R@5={mr5:.3f}  Δ={d_r5:+.4f} p={p_r5:.3f}    MRR={mmr:.3f}  Δ={d_mrr:+.4f} p={p_mrr:.3f}")

        out[label] = {n: rows_per_format[n] for n in FORMATS}

    with open(DATA / "mpnet_significance.json", "w") as f:
        json.dump(out, f)
    print(f"\nSaved → {DATA / 'mpnet_significance.json'}")


if __name__ == "__main__":
    main()
