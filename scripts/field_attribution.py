"""
Per-field per-category attribution.

For each ablation step (M1->M2, M2->M3, ..., M6->M7) and each query
category, compute the marginal Recall@5 gain. Heatmap-shaped output.
This shows which fields contribute to which query types — the schema
isn't a bag of fields, it's a functional decomposition.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.corpus import load_corpus
from benchmark.retrieval import get_retriever
from benchmark.enterprise_eval import get_format_modules, CATEGORIES
from benchmark.metrics import recall_at_k, mrr_at_k, paired_bootstrap_p

LEVELS = ["rods_m1", "rods_m2", "rods_m3", "rods_m4", "rods_m5", "rods_m6", "rods_m7"]
FIELDS = {
    "rods_m1": "L1 (headings)",
    "rods_m2": "+ Summary",
    "rods_m3": "+ Type",
    "rods_m4": "+ Entities",
    "rods_m5": "+ Aliases",
    "rods_m6": "+ Status/Date/Owner",
    "rods_m7": "+ Related",
}

DATA = Path(__file__).parent.parent / "data"


def main():
    documents = load_corpus()
    fmt_mods = dict(get_format_modules(LEVELS))
    r = get_retriever("hybrid")
    if hasattr(r, "_dense"):
        r._dense._model.encode(["warmup"], show_progress_bar=False)

    # Combined queries for tightest stats
    queries = []
    for path in [DATA / "enterprise_queries.json", DATA / "enterprise_queries_heldout.json"]:
        with open(path) as f:
            queries.extend(json.load(f)["queries"])
    print(f"Eval on {len(queries)} combined queries\n")

    # For each level, evaluate per-query R@5 against L1 baseline (= rods_m1).
    per_query_recall = {}
    for level_name in LEVELS:
        chunks = fmt_mods[level_name](documents)
        retriever = get_retriever("hybrid")
        idx = retriever.index(chunks)
        per_q = []
        for q in queries:
            targets = set(q["target_sections"])
            results = idx.search(q["question"], top_k=20)
            seen, seen_set = [], set()
            for chunk, _ in results:
                if chunk.source_section not in seen_set:
                    seen.append(chunk.source_section); seen_set.add(chunk.source_section)
            r5 = len(set(seen[:5]) & targets) / len(targets) if targets else 0.0
            mrr = mrr_at_k(seen, targets, 10)
            per_q.append({"category": q["category"], "r5": r5, "mrr": mrr})
        per_query_recall[level_name] = per_q

    # Marginal gain per category: M_{k} - M_{k-1}
    print(f"\n{'Category':<24}  {'M2-M1':>9}  {'M3-M2':>9}  {'M4-M3':>9}  {'M5-M4':>9}  {'M6-M5':>9}  {'M7-M6':>9}")
    print("  " + "─" * 90)
    for cat in CATEGORIES:
        cat_idx = [i for i, q in enumerate(queries) if q["category"] == cat]
        cells = []
        for prev_lvl, cur_lvl in zip(LEVELS, LEVELS[1:]):
            prev_r5 = [per_query_recall[prev_lvl][i]["r5"] for i in cat_idx]
            cur_r5 = [per_query_recall[cur_lvl][i]["r5"] for i in cat_idx]
            mean_diff = sum(c - p for c, p in zip(cur_r5, prev_r5)) / len(cat_idx)
            diff, p = paired_bootstrap_p(cur_r5, prev_r5, n_boot=1000)
            sig = "*" if p < 0.10 else " "
            cells.append(f"{mean_diff:>+8.3f}{sig}")
        print(f"  {cat:<22}    " + "  ".join(cells))

    # Cumulative R@5: M_k - M_1 (= against L1 baseline)
    print(f"\n=== CUMULATIVE GAIN vs L1 (M1), Recall@5 ===")
    print(f"{'Category':<24}  {'M2-M1':>9}  {'M3-M1':>9}  {'M4-M1':>9}  {'M5-M1':>9}  {'M6-M1':>9}  {'M7-M1':>9}")
    print("  " + "─" * 90)
    base = LEVELS[0]
    for cat in CATEGORIES:
        cat_idx = [i for i, q in enumerate(queries) if q["category"] == cat]
        cells = []
        for cur_lvl in LEVELS[1:]:
            base_r5 = [per_query_recall[base][i]["r5"] for i in cat_idx]
            cur_r5 = [per_query_recall[cur_lvl][i]["r5"] for i in cat_idx]
            mean_diff = sum(c - p for c, p in zip(cur_r5, base_r5)) / len(cat_idx)
            diff, p = paired_bootstrap_p(cur_r5, base_r5, n_boot=1000)
            sig = "**" if p < 0.05 else ("* " if p < 0.10 else "  ")
            cells.append(f"{mean_diff:>+7.3f}{sig}")
        print(f"  {cat:<22}    " + "  ".join(cells))

    # Cumulative MRR@10 (more sensitive than R@5 once R@5 saturates)
    print(f"\n=== CUMULATIVE GAIN vs L1 (M1), MRR@10 ===")
    print(f"{'Category':<24}  {'M2-M1':>9}  {'M3-M1':>9}  {'M4-M1':>9}  {'M5-M1':>9}  {'M6-M1':>9}  {'M7-M1':>9}")
    print("  " + "─" * 90)
    for cat in CATEGORIES:
        cat_idx = [i for i, q in enumerate(queries) if q["category"] == cat]
        cells = []
        for cur_lvl in LEVELS[1:]:
            base_mrr = [per_query_recall[base][i]["mrr"] for i in cat_idx]
            cur_mrr = [per_query_recall[cur_lvl][i]["mrr"] for i in cat_idx]
            mean_diff = sum(c - p for c, p in zip(cur_mrr, base_mrr)) / len(cat_idx)
            diff, p = paired_bootstrap_p(cur_mrr, base_mrr, n_boot=1000)
            sig = "**" if p < 0.05 else ("* " if p < 0.10 else "  ")
            cells.append(f"{mean_diff:>+7.3f}{sig}")
        print(f"  {cat:<22}    " + "  ".join(cells))

    # Save raw
    with open(DATA / "field_attribution.json", "w") as f:
        json.dump({lvl: per_query_recall[lvl] for lvl in LEVELS}, f)


if __name__ == "__main__":
    main()
