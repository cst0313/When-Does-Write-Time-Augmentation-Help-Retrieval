"""
Schema-generation cost curve.

For each RODS M-level, measure (a) tokens added per section vs L1
baseline, and (b) Recall@5 / MRR@10 gain. Output a small table that
can be plotted.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.corpus import load_corpus
from benchmark.retrieval import get_retriever
from benchmark.metrics import recall_at_k, mrr_at_k, count_tokens_approx
from benchmark.enterprise_eval import load_queries, get_format_modules

LEVELS = ["rods_m0", "rods_m1", "rods_m2", "rods_m3",
          "rods_m4", "rods_m5", "rods_m6", "rods_m7"]


def evaluate(format_name, fn, documents, queries, retriever_type):
    chunks = fn(documents)
    retriever = get_retriever(retriever_type)
    index = retriever.index(chunks)
    per_q = []
    for q in queries:
        targets = set(q["target_sections"])
        results = index.search(q["question"], top_k=20)
        seen, seen_set = [], set()
        for chunk, _ in results:
            if chunk.source_section not in seen_set:
                seen.append(chunk.source_section); seen_set.add(chunk.source_section)
            if len(seen) >= 20: break
        recall_5 = len(set(seen[:5]) & targets) / len(targets) if targets else 0.0
        per_q.append({"recall@5": recall_5, "mrr@10": mrr_at_k(seen, targets, 10)})
    avg_tokens = sum(count_tokens_approx(c.text) for c in chunks) / len(chunks)
    return per_q, avg_tokens


def main():
    documents = load_corpus()
    fmt_mods = dict(get_format_modules(LEVELS))

    r = get_retriever("hybrid")
    if hasattr(r, "_dense"):
        r._dense._model.encode(["warmup"], show_progress_bar=False)

    rows = []
    # Combined: original + held-out for tighter estimates
    queries_orig = load_queries()
    with open(Path(__file__).parent.parent / "data" / "enterprise_queries_heldout.json") as f:
        queries_held = json.load(f)["queries"]
    queries_combined = queries_orig + queries_held
    print(f"Eval on {len(queries_combined)} combined queries (original + held-out)")

    baseline_recall = None
    baseline_mrr = None
    baseline_tokens = None

    print(f"\n  {'Level':<8} {'tok/sec':>8} {'Δ tok':>7} {'R@5':>8} {'Δ R@5':>8} {'MRR':>8} {'Δ MRR':>8}")
    print("  " + "─" * 70)
    for level_name in LEVELS:
        if level_name not in fmt_mods: continue
        per_q, tokens = evaluate(level_name, fmt_mods[level_name], documents, queries_combined, "hybrid")
        recall = sum(p["recall@5"] for p in per_q) / len(per_q)
        mrr = sum(p["mrr@10"] for p in per_q) / len(per_q)
        if level_name == "rods_m1":
            baseline_recall = recall
            baseline_mrr = mrr
            baseline_tokens = tokens
        d_tok = tokens - baseline_tokens if baseline_tokens else 0
        d_recall = recall - baseline_recall if baseline_recall else 0
        d_mrr = mrr - baseline_mrr if baseline_mrr else 0
        print(f"  {level_name:<8} {tokens:>8.1f} {d_tok:>+7.1f} {recall:>8.3f} {d_recall:>+8.3f} {mrr:>8.3f} {d_mrr:>+8.3f}")
        rows.append({
            "level": level_name, "tokens_per_section": round(tokens, 1),
            "delta_tokens_vs_M1": round(d_tok, 1),
            "recall@5": round(recall, 4), "delta_recall@5_vs_M1": round(d_recall, 4),
            "mrr@10": round(mrr, 4), "delta_mrr@10_vs_M1": round(d_mrr, 4),
        })
    out = Path(__file__).parent.parent / "data" / "rods_cost_curve.tsv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
