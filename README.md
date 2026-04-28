# Released artefacts: A Divergence Diagnostic for RAG Pipelines

This archive contains the diagnostic harness, the held-out enterprise
benchmark, and the per-query rank traces referenced in the paper:

  *When Does Write-Time Augmentation Help Retrieval?
   A Divergence Diagnostic for RAG Pipelines* (Chang, 2026)

The PDF is included as `findings.pdf`.

---

## Layout

```
divergence_diagnostic_release/
├── findings.pdf                   # the paper
├── refs.bib                       # bibliography
├── scripts/                       # diagnostic harness + reproducible analyses
└── data/                          # per-query rank traces + result JSONs
└── benchmark_data/                # the held-out enterprise benchmark
```

---

## scripts/  — diagnostic harness

| Script                          | What it computes                                                  | Section in paper |
|---------------------------------|-------------------------------------------------------------------|------------------|
| `divergence_predictor.py`       | Jaccard divergence + per-query rank gain regression on each corpus | §7               |
| `diagnostic_validation.py`      | Closes the loop: 20-cell predicted-vs-observed table              | §7, Table 11     |
| `robustness_analysis.py`        | Holm-Bonferroni / BH-FDR / Cohen’s κ / split-half CV / power      | §7 (robustness)  |
| `mpnet_significance.py`         | Embedder swap (paired bootstrap on mpnet vs MiniLM)               | §6.7             |
| `reranker_test.py`              | Cross-encoder absorption test                                     | §6.6             |
| `enterprise_eval.py`            | Stratified enterprise eval driver (in-dist + held-out)            | §6.1, §6.2       |
| `field_attribution.py`          | Per-category MRR@10 attribution heatmap                           | §6.6, Figure 3   |
| `cost_curve.py`                 | Schema cost curve M0 → M7                                         | §6.5             |
| `paired_test_inline.py`         | Paired-bootstrap p-values inline                                  | §3, §6.2         |
| `fill_survival_table.py`        | Computes Table 10 (What survives) cells                           | §6.8             |
| `metrics.py`                    | Recall@K, MRR@K, bootstrap_ci, paired_bootstrap_p                 | §3               |

To run any script: `python -m scripts.<script_name>` (or `python scripts/<script>.py`
after adjusting imports). Most scripts read from `data/` and `benchmark_data/`.

---

## benchmark_data/  — the held-out enterprise benchmark

| File                              | Contents                                                         |
|-----------------------------------|------------------------------------------------------------------|
| `enterprise_queries.json`         | 80 in-distribution queries (8 categories × 10), with target_sections |
| `enterprise_queries_heldout.json` | 80 held-out queries authored without reference to the schema     |

Each entry: `{id, question, target_sections, category}`.

The 12-document / 72-section corpus they point at lives in the parent
project directory (`accelerated_md/data/wiki/...`). It is not bundled
here; the JSON files reference sections by stable `full_id`.

---

## data/  — per-query rank traces and result JSONs

| File                              | Contents                                                                    |
|-----------------------------------|-----------------------------------------------------------------------------|
| `divergence_predictor.json`       | Per-corpus / per-format: divergences, ranks (per query), gains_vs_L1. **The raw object behind §7 and Figure 5.** |
| `diagnostic_validation.json`      | 20 (corpus × format) calibration cells: β, p, R², mean div, mean gain      |
| `diagnostic_subsample_50.json`    | 1000 50-query bootstrap subsamples: distribution of β and p (the recipe check) |
| `reranker_results.json`           | Per-query R@5 / MRR@10 before and after cross-encoder rerank (4 formats)   |
| `mpnet_significance.json`         | Per-query R@5 / MRR@10 under all-mpnet-base-v2 (in-dist + held-out)        |
| `field_attribution.json`          | Per-query metrics for each ablation level (M0–M7), used by Figure 3        |

---

## Reproducing key paper results

| Result                                  | Command                                                  |
|-----------------------------------------|----------------------------------------------------------|
| Diagnostic regression + 16/20 calibration | `python -m benchmark.diagnostic_validation`            |
| Robustness checks (Holm-Bonferroni, kappa, split-half CV, power, ROC) | `python -m benchmark.robustness_analysis` |
| Cross-encoder absorption (§6.6)          | `python -m benchmark.reranker_test`                    |
| Embedder swap (§6.7)                     | `python -m benchmark.mpnet_significance`               |
| Cost curve (§6.5)                        | `python -m benchmark.cost_curve`                       |

(Paths assume the scripts are placed back into `accelerated_md/benchmark/`;
the standalone bundle uses `python scripts/<name>.py` instead.)

---

## License & contact

Released for academic use; please cite the paper if you build on the
diagnostic.
