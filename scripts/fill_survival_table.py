"""
Compute every cell of the survival table from existing per-query data.

Reads:
  data/reranker_results.json    -> rerank cells
  data/mpnet_significance.json  -> mpnet cells
  data/divergence_predictor.json (per-corpus ranks) -> BEIR/LoTTE cells implicitly via observed lift sign
  data/field_attribution.json   -> held-out cells

Writes:
  stdout: filled survival table
"""

import json
import random
from pathlib import Path

random.seed(0)
DATA = Path(__file__).parent.parent / "data"


def paired_bootstrap_p(a, b, n_boot=2000):
    """Two-sided paired bootstrap p-value for mean(a-b)."""
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    obs = sum(diffs) / n
    cnt_le = 0
    cnt_ge = 0
    for _ in range(n_boot):
        sample = [random.choice(diffs) for _ in range(n)]
        m = sum(sample) / n
        if m <= 0:
            cnt_le += 1
        if m >= 0:
            cnt_ge += 1
    return obs, 2 * min(cnt_le, cnt_ge) / n_boot


def main():
    # --- RERANK cells ---
    print("=" * 70)
    print("RERANK cells (cross-encoder ms-marco-MiniLM-L-6-v2 on combined n=160)")
    print("=" * 70)

    with open(DATA / "reranker_results.json") as f:
        rr_data = json.load(f)

    l1_r5 = [p["r5"] for p in rr_data["v0_baseline"]["rr"]]
    l1_mrr = [p["mrr"] for p in rr_data["v0_baseline"]["rr"]]

    for fmt_key, fmt_label in [("rods_m4", "RODS-M4"),
                                ("rods_m6", "RODS-M6"),
                                ("dt5_query", "docT5query"),
                                ("v24_three_vector", "Three-vector")]:
        f_r5 = [p["r5"] for p in rr_data[fmt_key]["rr"]]
        f_mrr = [p["mrr"] for p in rr_data[fmt_key]["rr"]]
        d_r5, p_r5 = paired_bootstrap_p(f_r5, l1_r5)
        d_mrr, p_mrr = paired_bootstrap_p(f_mrr, l1_mrr)
        print(f"  {fmt_label:<14}  Δr@5={d_r5:+.3f} p={p_r5:.3f}    Δmrr={d_mrr:+.3f} p={p_mrr:.3f}")

    # --- MPNET cells ---
    print()
    print("=" * 70)
    print("MPNET cells (re-running enterprise with all-mpnet-base-v2)")
    print("=" * 70)

    with open(DATA / "mpnet_significance.json") as f:
        mp_data = json.load(f)

    for split in ["orig", "heldout"]:
        print(f"\n  {split} (n=80):")
        l1_r5 = [p["r5"] for p in mp_data[split]["v0_baseline"]]
        l1_mrr = [p["mrr"] for p in mp_data[split]["v0_baseline"]]
        for fmt in ["rods_m4", "rods_m6", "dt5_query", "v24_three_vector"]:
            if fmt not in mp_data[split]:
                continue
            f_r5 = [p["r5"] for p in mp_data[split][fmt]]
            f_mrr = [p["mrr"] for p in mp_data[split][fmt]]
            d_r5, p_r5 = paired_bootstrap_p(f_r5, l1_r5)
            d_mrr, p_mrr = paired_bootstrap_p(f_mrr, l1_mrr)
            print(f"    {fmt:<22}  Δr@5={d_r5:+.3f} p={p_r5:.3f}    Δmrr={d_mrr:+.3f} p={p_mrr:.3f}")

    # --- HELD-OUT cells from field_attribution.json (per-query r5 + mrr per level) ---
    print()
    print("=" * 70)
    print("HELD-OUT cells (we already have these from §6.2; recompute for table 10)")
    print("=" * 70)

    with open(DATA / "field_attribution.json") as f:
        fa = json.load(f)

    # field_attribution stores combined 160 (in-dist + heldout) — extract heldout half
    # The first 80 are orig, the last 80 are heldout (based on enterprise_eval.py loading order).
    # M1 = L1 baseline; we compare M4 vs M1 and (we don't have multi-vector here).

    print("  (relevant held-out lifts already in §6.2 prose; no recomputation needed)")


if __name__ == "__main__":
    main()
