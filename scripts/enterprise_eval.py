"""
Enterprise-knowledge stratified evaluation for RODS.

80 queries × 8 categories (fact / decision / why / multi-hop / temporal /
procedural / ambiguous-entity / open-ended summary). Stratified Recall@K
per category to identify where structured documents help.
"""

import csv
import json
import time
from pathlib import Path

from benchmark.corpus import load_corpus
from benchmark.retrieval import RetrieverType, get_retriever
from benchmark.metrics import recall_at_k, mrr_at_k, count_tokens_approx, bootstrap_ci, paired_bootstrap_p

DATA = Path(__file__).parent.parent / "data"
QUERIES = DATA / "enterprise_queries.json"
RESULTS = DATA / "enterprise_results.tsv"

CATEGORIES = ["fact_lookup", "decision_lookup", "why_rationale", "multi_hop",
              "temporal_superseded", "procedural", "ambiguous_entity",
              "open_ended_summary"]


def load_queries():
    with open(QUERIES) as f:
        return json.load(f)["queries"]


def get_format_modules(format_names):
    out = []
    import importlib
    for n in format_names:
        try:
            mod = importlib.import_module(f"benchmark.formats.{n}")
            out.append((mod.FORMAT_NAME, mod.corpus_to_chunks))
        except (ImportError, AttributeError) as e:
            print(f"  (skip {n}: {e})")
    return out


def evaluate_format(name, chunks, queries, retriever_type, k_values=(3, 5, 10)):
    retriever = get_retriever(retriever_type)
    index = retriever.index(chunks)
    rows = []  # per-query metrics
    for q in queries:
        targets = set(q["target_sections"])
        results = index.search(q["question"], top_k=min(50, len(chunks)))
        # Dedupe by source_section (chunks may have multiple variants per section)
        seen = []
        seen_set = set()
        for chunk, _ in results:
            if chunk.source_section not in seen_set:
                seen.append(chunk.source_section)
                seen_set.add(chunk.source_section)
            if len(seen) >= 50:
                break
        # Recall@K (counted against the set of target sections)
        per_q = {"qid": q["id"], "category": q["category"], "n_targets": len(targets)}
        for k in k_values:
            top_k = set(seen[:k])
            per_q[f"recall@{k}"] = len(top_k & targets) / len(targets) if targets else 0.0
        per_q["mrr@10"] = mrr_at_k(seen, targets, 10)
        rows.append(per_q)
    return rows


def aggregate(rows, by_category=True, with_ci=False):
    overall = {}
    for k in (3, 5, 10):
        vals = [r[f"recall@{k}"] for r in rows]
        if with_ci:
            mean, lo, hi = bootstrap_ci(vals)
            overall[f"recall@{k}"] = mean
            overall[f"recall@{k}_ci"] = (lo, hi)
        else:
            overall[f"recall@{k}"] = sum(vals) / len(vals)
    mrr_vals = [r["mrr@10"] for r in rows]
    if with_ci:
        mean, lo, hi = bootstrap_ci(mrr_vals)
        overall["mrr@10"] = mean
        overall["mrr@10_ci"] = (lo, hi)
    else:
        overall["mrr@10"] = sum(mrr_vals) / len(mrr_vals)
    by_cat = {}
    if by_category:
        for cat in CATEGORIES:
            cat_rows = [r for r in rows if r["category"] == cat]
            if not cat_rows:
                continue
            by_cat[cat] = {
                f"recall@5": sum(r["recall@5"] for r in cat_rows) / len(cat_rows),
                "mrr@10": sum(r["mrr@10"] for r in cat_rows) / len(cat_rows),
                "n": len(cat_rows),
            }
    return overall, by_cat


def run(retriever_type: RetrieverType = "hybrid", queries_path: Path = None,
        with_ci: bool = True):
    print(f"\n{'═' * 100}")
    print(f"  ENTERPRISE STRATIFIED EVAL  ·  retriever: {retriever_type}  ·  CI: {with_ci}")
    print('═' * 100)

    documents = load_corpus()
    if queries_path is None:
        queries = load_queries()
    else:
        with open(queries_path) as f:
            queries = json.load(f)["queries"]
        print(f"  Using queries from {queries_path}")
    print(f"  Corpus: {len(documents)} docs · {sum(len(d.sections) for d in documents)} sections")
    print(f"  Queries: {len(queries)} stratified into {len(CATEGORIES)} categories\n")

    if retriever_type in ("dense", "hybrid"):
        r = get_retriever(retriever_type)
        if hasattr(r, "_dense"):
            r._dense._model.encode(["warmup"], show_progress_bar=False)

    # Compare RODS M0..M7 against baselines + docT5query
    formats = [
        "bl_fixed_chunks",     # L0
        "v0_baseline",         # L1 = heading-aware (raw markdown)
        "bl_semantic_chunks",  # L2
        "bl_summary_chunks",   # L3
        "dt5_query",           # docT5query baseline
        "rods_m0", "rods_m1", "rods_m2", "rods_m3",
        "rods_m4", "rods_m5", "rods_m6", "rods_m7",
        "v6c_colloquial",
        "v24_three_vector", "v71_v64_synonym_chunk",
    ]

    fmt_modules = get_format_modules(formats)

    print(f"  {'Format':<22} {'R@3':>5} {'R@5':>5} {'R@10':>5} {'MRR@10':>7}  Best categories")
    print("  " + "─" * 100)

    saved_rows = []
    for name, fn in fmt_modules:
        chunks = fn(documents)
        t0 = time.perf_counter()
        rows = evaluate_format(name, chunks, queries, retriever_type)
        dt = time.perf_counter() - t0
        overall, by_cat = aggregate(rows, with_ci=with_ci)
        n_chunks = len(chunks)
        # Format with CI if available
        def fmt(metric):
            mean = overall[metric]
            ci = overall.get(f"{metric}_ci")
            if ci:
                return f"{mean:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
            return f"{mean:.3f}"
        print(f"  {name:<22} R@3={fmt('recall@3')}  R@5={fmt('recall@5')}  "
              f"R@10={fmt('recall@10')}  MRR={fmt('mrr@10')}  "
              f"({dt:.1f}s, {n_chunks} chunks)")
        saved = {"format": name, "n_chunks": n_chunks, "ms_per_q": dt / len(queries) * 1000,
                 "per_query_recall@5": [r["recall@5"] for r in rows]}
        for k, v in overall.items():
            if "_ci" in k: saved[k] = v
            else: saved[k] = v
        for cat, m in by_cat.items():
            saved[f"R@5_{cat}"] = m["recall@5"]
        saved_rows.append(saved)

    # Save TSV
    with open(RESULTS, "w", newline="") as f:
        all_keys = sorted({k for r in saved_rows for k in r.keys()})
        # Put format/n_chunks first
        cols = ["format", "n_chunks", "ms_per_q",
                "recall@3", "recall@5", "recall@10", "mrr@10"] + \
               [k for k in all_keys if k.startswith("R@5_")]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in saved_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in cols})
    print(f"\n  Results saved → {RESULTS}\n")

    # Per-category leaderboard
    print(f"  Per-category Recall@5 leaderboard:\n")
    print(f"  {'Category':<22}  " + "  ".join(f"{f[:14]:>14}" for f, _ in fmt_modules[:6]))
    for cat in CATEGORIES:
        cells = []
        for name, _ in fmt_modules[:6]:
            sr = next((r for r in saved_rows if r["format"] == name), None)
            if sr and f"R@5_{cat}" in sr:
                cells.append(f"{sr[f'R@5_{cat}']:>14.3f}")
            else:
                cells.append(f"{'-':>14}")
        print(f"  {cat:<22}  " + "  ".join(cells))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dense", action="store_true")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--hyde", action="store_true", help="Use HyDE retriever")
    p.add_argument("--queries", type=str, default=None, help="Path to alt query file (e.g. heldout)")
    p.add_argument("--no-ci", action="store_true")
    args = p.parse_args()
    if args.hyde: rt = "hyde"
    elif args.hybrid: rt = "hybrid"
    elif args.dense: rt = "dense"
    else: rt = "tfidf"
    qpath = Path(args.queries) if args.queries else None
    run(retriever_type=rt, queries_path=qpath, with_ci=not args.no_ci)
