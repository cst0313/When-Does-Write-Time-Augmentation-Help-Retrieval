"""
Vocabulary-divergence predictor.

For each (query, target_section) pair across all corpora, compute three
divergence measures:
  - lex_divergence:  1 - Jaccard(query_tokens, doc_tokens)   ∈ [0, 1]
  - bm25_div:        normalised BM25 overlap (low = divergent)
  - emb_div:         1 - cosine_similarity(emb(q), emb(d))   ∈ [0, 2]

Then for each format (L1 baseline, RODS-M6, docT5query, three-vector,
multi-vector with synonyms), compute per-query rank improvement over L1
and show that GAIN is predictable from DIVERGENCE.

The hypothesis: write-time augmentation helps proportionally to how far
the query is from the document in vocabulary/embedding space. This is the
unifying empirical law behind the cross-corpus pattern.
"""

import json
import re
import sys
import math
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.corpus import load_corpus
from benchmark.retrieval import get_retriever
from benchmark.enterprise_eval import get_format_modules
from benchmark.metrics import bootstrap_ci

DATA = Path(__file__).parent.parent / "data"


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
_STOP = {"the","a","an","of","in","to","and","or","is","are","was","were","for",
         "with","by","on","at","as","this","that","it","be","been","from","but",
         "not","have","has","had","can","could","would","may","we","they","he",
         "she","i","you","their","his","her","its","also","such","more","most"}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP and len(t) >= 3]


def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def lex_divergence(query: str, doc_text: str) -> float:
    """1 - query_term_recall(query, doc).
    0 = every content word in the query appears in the doc;
    1 = no content word in the query appears in the doc.
    """
    qt = set(tokens(query))
    dt = set(tokens(doc_text))
    if not qt: return 0.0
    recall = len(qt & dt) / len(qt)
    return 1.0 - recall


def per_query_rank(retriever, queries_with_targets, top_k=20):
    """For each query, return rank of FIRST target section in top_k (or top_k+1 if not found)."""
    out = []
    for q, targets in queries_with_targets:
        results = retriever.search(q, top_k=top_k)
        seen = []
        seen_set = set()
        for chunk, _ in results:
            if chunk.source_section not in seen_set:
                seen.append(chunk.source_section)
                seen_set.add(chunk.source_section)
        target_set = set(targets)
        for r, sid in enumerate(seen, start=1):
            if sid in target_set:
                out.append(r)
                break
        else:
            out.append(top_k + 1)
    return out


def evaluate_corpus(corpus_label, queries_with_targets_and_docs, formats_to_test):
    """
    queries_with_targets_and_docs: list of (query_text, [target_section_id], target_doc_text)
    formats_to_test: list of (name, build_chunks_fn)
    Returns: dict {format_name: list of (lex_div, rank, gain_vs_L1)}
    """
    print(f"\n=== {corpus_label} ({len(queries_with_targets_and_docs)} queries) ===")
    queries_with_targets = [(q, ts) for q, ts, _ in queries_with_targets_and_docs]

    # Compute lex divergence for each query
    divergences = [lex_divergence(q, dt) for q, _, dt in queries_with_targets_and_docs]

    rank_per_format = {}
    for name, fn in formats_to_test:
        chunks = fn()
        retriever = get_retriever("hybrid")
        idx = retriever.index(chunks)
        rank_per_format[name] = per_query_rank(idx, queries_with_targets, top_k=20)

    # Compute gain vs L1 baseline per query (lower rank is better)
    if "L1" in rank_per_format:
        baseline_ranks = rank_per_format["L1"]
        per_format_gains = {}
        for name, ranks in rank_per_format.items():
            if name == "L1": continue
            gains = [b - r for b, r in zip(baseline_ranks, ranks)]  # positive = format BETTER than L1
            per_format_gains[name] = gains
    else:
        per_format_gains = {}

    return {
        "label": corpus_label,
        "divergences": divergences,
        "ranks": rank_per_format,
        "gains_vs_L1": per_format_gains,
    }


# Build queries + target docs for each corpus
def build_enterprise_data():
    """Combined original + held-out enterprise."""
    documents = load_corpus()
    sec_text = {s.full_id: s.content for d in documents for s in d.sections}
    out = []
    for path in [DATA / "enterprise_queries.json", DATA / "enterprise_queries_heldout.json"]:
        with open(path) as f:
            for q in json.load(f)["queries"]:
                ts = q["target_sections"]
                # Use first target's content for divergence calc (typical)
                doc_text = sec_text.get(ts[0], "") if ts else ""
                out.append((q["question"], ts, doc_text))
    return out


def build_beir_data(name):
    with open(DATA / f"beir_{name}.json") as f:
        d = json.load(f)
    doc_text = {str(c["_id"]): (c.get("title", "") + " " + c.get("text", "")) for c in d["corpus"]}
    out = []
    for q in d["queries"]:
        qid = q["_id"]
        if qid not in d["qrels"]: continue
        first_doc = next(iter(d["qrels"][qid].keys()))
        # target section_id is f"{name}/{first_doc}" since beir corpus loads with doc_id=name
        target = f"{name}/{first_doc}"
        out.append((q["text"], [target], doc_text.get(first_doc, "")))
    return out


def build_hotpot_data():
    with open(DATA / "hotpot_500.json") as f:
        d = json.load(f)
    title_to_text = {c["title"]: " ".join(c["sentences"]) for c in d["context"]}
    out = []
    for q in d["queries"]:
        # Need the title-id transformation
        sf = q.get("supporting_facts", [])
        # Get first supporting title's section_id
        if not sf: continue
        first_title = sf[0]["title"]
        section_id = first_title.lower().replace(" ", "_").replace("'", "").replace('"', "")[:60]
        target = f"hotpot/{section_id}"
        out.append((q["question"], [target], title_to_text.get(first_title, "")))
    return out[:300]  # cap for tractability


def main():
    # Warm dense
    r = get_retriever("hybrid")
    if hasattr(r, "_dense"):
        r._dense._model.encode(["warmup"], show_progress_bar=False)

    documents = load_corpus()

    # Set up format builders for the synthetic enterprise corpus
    fmt_mods = dict(get_format_modules(["v0_baseline", "rods_m4", "rods_m6", "v24_three_vector", "dt5_query"]))
    enterprise_formats = [
        ("L1", lambda: fmt_mods["v0_baseline"](documents)),
        ("RODS-M4", lambda: fmt_mods["rods_m4"](documents)),
        ("RODS-M6", lambda: fmt_mods["rods_m6"](documents)),
        ("docT5query", lambda: fmt_mods["dt5_query"](documents)),
        ("Three-vector", lambda: fmt_mods["v24_three_vector"](documents)),
    ]

    # Run on enterprise (combined 160 queries)
    ent_data = build_enterprise_data()
    print(f"Enterprise data: {len(ent_data)} queries")
    ent_result = evaluate_corpus("enterprise (combined 160)", ent_data, enterprise_formats)

    all_results = {"enterprise": {
        "divergences": ent_result["divergences"],
        "ranks": dict(ent_result["ranks"]),
        "gains_vs_L1": ent_result["gains_vs_L1"],
    }}

    # Run on BEIR + LoTTE corpora
    from benchmark.beir_eval import load_beir, build_caches, build_format_chunks
    beir_results = {}
    for ds_name in ["scifact", "nfcorpus", "fiqa", "lottewriting5k"]:
        print(f"\n=== Loading {ds_name} ===")
        try:
            beir_docs, beir_queries, beir_qrels = load_beir(ds_name)
        except Exception as e:
            print(f"  skip {ds_name}: {e}")
            continue
        beir_caches = build_caches(beir_docs, ds_name)
        sec_to_text = {f"{ds_name}/{c.section_id}": c.content for d in beir_docs for c in d.sections}
        d_data = []
        for q in beir_queries:
            if q["_id"] not in beir_qrels or not beir_qrels[q["_id"]]: continue
            first_doc = next(iter(beir_qrels[q["_id"]].keys()))
            target_id = f"{ds_name}/{first_doc}"
            d_data.append((q["text"], [target_id], sec_to_text.get(target_id, "")))
        # Cap to 300 queries for tractability
        d_data = d_data[:300]
        print(f"  {ds_name} data: {len(d_data)} queries")
        ds_formats = [
            ("L1", lambda d=beir_docs, c=beir_caches: build_format_chunks("v0_baseline", d, c)),
            ("RODS-M4", lambda d=beir_docs, c=beir_caches: build_format_chunks("rods_m4", d, c)),
            ("RODS-M6", lambda d=beir_docs, c=beir_caches: build_format_chunks("rods_m6", d, c)),
            ("docT5query", lambda d=beir_docs, c=beir_caches: build_format_chunks("dt5_query", d, c)),
            ("Three-vector", lambda d=beir_docs, c=beir_caches: build_format_chunks("v24_three_vector", d, c)),
        ]
        result = evaluate_corpus(f"{ds_name} ({len(d_data)})", d_data, ds_formats)
        beir_results[ds_name] = result
        all_results[ds_name] = {
            "divergences": result["divergences"],
            "ranks": dict(result["ranks"]),
            "gains_vs_L1": result["gains_vs_L1"],
        }
    sf_result = beir_results.get("scifact")

    # Save raw results
    out_path = DATA / "divergence_predictor.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print binned tables for all corpora
    bins = [(0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]
    corpora_to_show = [("ENTERPRISE", ent_result)] + [(k.upper(), v) for k, v in beir_results.items()]
    for label, result in corpora_to_show:
        if result is None: continue
        print(f"\n=== {label}: Gain vs Lex Divergence (1 - query-term recall in target) ===")
        div = result["divergences"]
        names = list(result["gains_vs_L1"].keys())
        print(f"{'Bin (lex_div)':<18} {'n':>4}  " + "  ".join(f"{name:>14}" for name in names))
        for lo, hi in bins:
            idxs = [i for i, d in enumerate(div) if lo <= d < hi]
            n = len(idxs)
            if n == 0:
                print(f"  [{lo:.2f},{hi:.2f}]      {n:>4}"); continue
            cells = [f"{sum(result['gains_vs_L1'][nm][i] for i in idxs) / n:>+14.2f}" for nm in names]
            print(f"  [{lo:.2f},{hi:.2f}]      {n:>4}  " + "  ".join(cells))
        # Mean
        n = len(div)
        cells = [f"{sum(result['gains_vs_L1'][nm]) / n:>+14.2f}" for nm in names]
        print(f"  ALL              {n:>4}  " + "  ".join(cells))

    # Linear regression: gain = α + β × divergence per format
    print(f"\n=== LINEAR REGRESSION: gain ~ α + β × divergence (per corpus, per format) ===")
    print(f"  (POSITIVE β = augmentation helps more at higher divergence)")
    print(f"  {'Format':<14}  {'corpus':<14}  {'β':>9}  {'α':>9}  {'R²':>7}  {'p (β!=0)':>10}")
    import math
    regression_records = []
    corpora_for_reg = [("enterprise", ent_result)] + [(k, v) for k, v in beir_results.items()]
    for label, result in corpora_for_reg:
        div = result["divergences"]
        for fmt_name, gains in result["gains_vs_L1"].items():
            n = len(div)
            mean_x = sum(div) / n
            mean_y = sum(gains) / n
            num = sum((div[i] - mean_x) * (gains[i] - mean_y) for i in range(n))
            den = sum((div[i] - mean_x) ** 2 for i in range(n))
            if den == 0: continue
            beta = num / den
            alpha = mean_y - beta * mean_x
            ss_tot = sum((g - mean_y) ** 2 for g in gains)
            ss_res = sum((gains[i] - (alpha + beta * div[i])) ** 2 for i in range(n))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            # Permutation test for β
            import random
            rng = random.Random(42)
            extreme = 0
            for _ in range(2000):
                shuffled = list(gains); rng.shuffle(shuffled)
                m_y = sum(shuffled) / n
                num2 = sum((div[i] - mean_x) * (shuffled[i] - m_y) for i in range(n))
                if abs(num2 / den) >= abs(beta): extreme += 1
            p = extreme / 2000
            sig = "**" if p < 0.05 else "  "
            print(f"  {sig}{fmt_name:<14}  {label:<14}  {beta:>+9.3f}  {alpha:>+9.3f}  {r2:>7.3f}  {p:>10.3f}")
            regression_records.append({"format": fmt_name, "corpus": label, "beta": beta,
                                        "alpha": alpha, "r2": r2, "p": p, "n": n})

    # Pooled regression across all corpora (with corpus as fixed effect / mean-centered per corpus)
    print(f"\n=== POOLED REGRESSION (all corpora, mean-centred per corpus) ===")
    print(f"  {'Format':<14}  {'pooled β':>9}  {'R²':>7}  {'p (β!=0)':>10}  {'n':>6}")
    for fmt_name in ["RODS-M4", "RODS-M6", "docT5query", "Three-vector"]:
        all_div = []
        all_gain = []
        for label, result in corpora_for_reg:
            if fmt_name not in result["gains_vs_L1"]: continue
            div_c = result["divergences"]
            gain_c = result["gains_vs_L1"][fmt_name]
            mean_d = sum(div_c) / len(div_c)
            mean_g = sum(gain_c) / len(gain_c)
            for d_, g_ in zip(div_c, gain_c):
                all_div.append(d_ - mean_d)
                all_gain.append(g_ - mean_g)
        n = len(all_div)
        if n == 0: continue
        mean_x = sum(all_div) / n
        mean_y = sum(all_gain) / n
        num = sum((all_div[i] - mean_x) * (all_gain[i] - mean_y) for i in range(n))
        den = sum((all_div[i] - mean_x) ** 2 for i in range(n))
        if den == 0: continue
        beta = num / den
        ss_tot = sum((g - mean_y) ** 2 for g in all_gain)
        ss_res = sum((all_gain[i] - (mean_y + beta * (all_div[i] - mean_x))) ** 2 for i in range(n))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # Permutation
        import random
        rng = random.Random(42)
        extreme = 0
        for _ in range(2000):
            shuffled = list(all_gain); rng.shuffle(shuffled)
            num2 = sum((all_div[i] - mean_x) * (shuffled[i] - mean_y) for i in range(n))
            if abs(num2 / den) >= abs(beta): extreme += 1
        p = extreme / 2000
        sig = "**" if p < 0.05 else "  "
        print(f"  {sig}{fmt_name:<14}  {beta:>+9.3f}  {r2:>7.3f}  {p:>10.3f}  {n:>6}")


if __name__ == "__main__":
    main()
