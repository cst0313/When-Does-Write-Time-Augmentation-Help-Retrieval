"""Generate the three figures referenced by main.tex (Plotly, PNG output).

Every value is computed directly from the JSON in ../data/. No invented
numbers, no jitter, no smoothing. Verified against the paper:

  - reranker_bars.png  : numbers match main.tex Table 2 to 3dp
  - per_category.png   : M1 column is identically 0 by construction
  - divergence_law.png : per-format OLS slope, intercept, R^2,
                          permutation p match scripts/divergence_predictor.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FONT = "Arial, Helvetica, sans-serif"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def sem(xs):
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (var / n) ** 0.5


def ols_fit(x, y):
    """Return (slope, intercept, r2, p_perm) — same formulae as
    scripts/divergence_predictor.py (seed=42, 2000 permutations)."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    if den == 0:
        return 0.0, my, 0.0, 1.0
    slope = num / den
    intercept = my - slope * mx
    ss_tot = sum((g - my) ** 2 for g in y)
    ss_res = sum((y[i] - (intercept + slope * x[i])) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rng = random.Random(42)
    extreme = 0
    for _ in range(2000):
        sh = list(y)
        rng.shuffle(sh)
        m_y = sum(sh) / n
        num2 = sum((x[i] - mx) * (sh[i] - m_y) for i in range(n))
        if abs(num2 / den) >= abs(slope):
            extreme += 1
    return slope, intercept, r2, extreme / 2000


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def write_png(fig, name, width, height, scale=3):
    """Write both PNG (high-DPI raster) and PDF (vector) variants so
    \\includegraphics{figures/<stem>.pdf|.png} both resolve."""
    stem = Path(name).stem
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.write_image(str(png), format="png", width=width, height=height,
                    scale=scale, engine="kaleido")
    fig.write_image(str(pdf), format="pdf", width=width, height=height,
                    engine="kaleido")
    print(f"wrote {png} + {pdf.name} ({width}x{height})")


# -------------------------------------------------------------------- figure 1
def fig_reranker_bars():
    d = json.loads((DATA / "reranker_results.json").read_text())
    order = [
        ("v0_baseline",      "L1<br>(heading-aware)"),
        ("rods_m4",          "RODS-M4"),
        ("rods_m6",          "RODS-M6"),
        ("dt5_query",        "docT5query"),
        ("v24_three_vector", "Three-vector"),
    ]
    labels = [lab for _, lab in order]

    series = [
        ("R@5 (hybrid)",                  "no_rr", "r5",  "#3b6fb6"),
        ("R@5 (+ cross-encoder rerank)",  "rr",    "r5",  "#e67c2c"),
        ("MRR@10 (+ rerank)",             "rr",    "mrr", "#3fa15a"),
    ]

    fig = go.Figure()
    n_groups = len(series)
    bargap = 0.30
    bargroupgap = 0.06
    bar_slot = (1 - bargap) / n_groups
    label_annotations = []
    for i, (name, bucket, metric, color) in enumerate(series):
        ys = [mean(r[metric] for r in d[k][bucket]) for k, _ in order]
        es = [sem(r[metric]  for r in d[k][bucket]) for k, _ in order]
        fig.add_bar(
            x=labels, y=ys, name=name,
            marker=dict(color=color, line=dict(color="white", width=0)),
            error_y=dict(type="data", array=es, thickness=1.2, width=5,
                         color="#222"),
        )
        offset_x = (i - (n_groups - 1) / 2) * bar_slot
        for j, (v, e) in enumerate(zip(ys, es)):
            label_annotations.append(dict(
                x=j + offset_x, y=v + e + 0.006,
                xref="x", yref="y",
                text=f"<b>{v:.3f}</b>", showarrow=False,
                font=dict(size=14, family=FONT, color="#111"),
                xanchor="center", yanchor="bottom",
            ))

    fig.update_layout(
        barmode="group",
        bargap=bargap,
        bargroupgap=bargroupgap,
        font=dict(family=FONT, size=15, color="#222"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text="<b>Cross-encoder reranking on the combined enterprise set "
                 "(<i>n</i>=160)</b>",
            x=0.02, xanchor="left", y=0.97, yanchor="top",
            font=dict(size=18, color="#111"),
        ),
        yaxis=dict(
            title=dict(text="Score", font=dict(size=16)),
            range=[0.78, 1.005],
            gridcolor="#ececec", gridwidth=1,
            zeroline=False,
            showline=False, ticks="outside", tickcolor="#888",
            tickfont=dict(size=14),
        ),
        xaxis=dict(
            showline=True, linecolor="#888", linewidth=1,
            ticks="", tickfont=dict(size=15),
        ),
        annotations=label_annotations,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.06,
            xanchor="right",  x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
        ),
        margin=dict(l=80, r=30, t=120, b=70),
    )
    write_png(fig, "reranker_bars.png", width=1100, height=560)


# -------------------------------------------------------------------- figure 2
def fig_per_category():
    fa = json.loads((DATA / "field_attribution.json").read_text())

    cat_order = []
    seen = set()
    for r in fa["rods_m1"]:
        if r["category"] not in seen:
            seen.add(r["category"])
            cat_order.append(r["category"])
    cat_label = {
        "fact_lookup":         "fact lookup",
        "decision_lookup":     "decision lookup",
        "why_rationale":       "why / rationale",
        "multi_hop":           "multi-hop",
        "temporal_superseded": "temporal / superseded",
        "procedural":          "procedural",
        "ambiguous_entity":    "ambiguous entity",
        "open_ended_summary":  "open-ended summary",
    }
    schemas = ["rods_m1", "rods_m2", "rods_m3", "rods_m4",
               "rods_m5", "rods_m6", "rods_m7"]
    schema_label = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]

    def per_cat_mrr(rows):
        bucket = defaultdict(list)
        for r in rows:
            bucket[r["category"]].append(r["mrr"])
        return {c: sum(v) / len(v) for c, v in bucket.items()}

    base = per_cat_mrr(fa["rods_m1"])
    z = []
    for cat in cat_order:
        row = []
        for s in schemas:
            cur = per_cat_mrr(fa[s])
            row.append(cur[cat] - base[cat])
        z.append(row)

    vmax = max(abs(v) for row in z for v in row)
    vmax = max(vmax, 0.05)

    # Render annotations manually so we can choose text colour per cell
    # for legibility against the underlying RdBu cell.
    annotations = []
    for i, cat in enumerate(cat_order):
        for j, s in enumerate(schema_label):
            v = z[i][j]
            txt = f"{v:+.3f}" if abs(v) >= 0.0005 else "0.000"
            # white text only on the deepest cells; black/dark on light/mid.
            light = abs(v) > 0.65 * vmax
            color = "white" if light else "#1a1a1a"
            annotations.append(dict(
                x=s, y=cat_label[cat], text=f"<b>{txt}</b>", showarrow=False,
                font=dict(size=14, family=FONT, color=color),
            ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=schema_label,
        y=[cat_label[c] for c in cat_order],
        colorscale=[
            [0.0,  "#2166ac"],
            [0.25, "#67a9cf"],
            [0.5,  "#f7f7f7"],
            [0.75, "#ef8a62"],
            [1.0,  "#b2182b"],
        ],
        zmid=0, zmin=-vmax, zmax=vmax,
        xgap=2, ygap=2,
        colorbar=dict(
            title=dict(text="Δ MRR@10<br>vs. M1", side="top",
                       font=dict(size=14)),
            thickness=14, len=0.75, outlinewidth=0,
            tickfont=dict(size=13),
            x=1.02, xanchor="left",
        ),
        hovertemplate=(
            "schema=%{x}<br>category=%{y}<br>"
            "ΔMRR@10=%{z:+.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        annotations=annotations,
        font=dict(family=FONT, size=14, color="#222"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text="<b>Per-category Δ MRR@10 by schema cut "
                 "(M1 column is 0 by construction; <i>n</i>=20 per category)</b>",
            x=0.02, xanchor="left", y=0.97, yanchor="top",
            font=dict(size=18, color="#111"),
        ),
        xaxis=dict(
            title=dict(text="Schema cut (cumulative)", font=dict(size=16)),
            side="bottom", showgrid=False,
            showline=False,
            ticks="", tickfont=dict(size=15),
        ),
        yaxis=dict(
            title=dict(text="Query category", font=dict(size=16)),
            autorange="reversed",
            showgrid=False,
            showline=False,
            ticks="", tickfont=dict(size=15),
        ),
        margin=dict(l=220, r=140, t=90, b=80),
    )
    write_png(fig, "per_category.png", width=1200, height=620)


# -------------------------------------------------------------------- figure 3
def fig_divergence_law():
    dp = json.loads((DATA / "divergence_predictor.json").read_text())

    formats = ["RODS-M4", "RODS-M6", "docT5query", "Three-vector"]
    colors = {
        "RODS-M4":      "#3b6fb6",
        "RODS-M6":      "#3fa15a",
        "docT5query":   "#8e6dbf",
        "Three-vector": "#d6422a",
    }

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=(
            "Enterprise  (augmentation-friendly)",
            "SciFact  (augmentation-neutral)",
        ),
        horizontal_spacing=0.10,
    )
    # Plotly subplot_titles are rendered as figure-level annotations; bump
    # their font size after creation.
    for ann in fig.layout.annotations:
        ann.font = dict(size=16, family=FONT, color="#111")

    xs_line = [i / 100 for i in range(101)]

    panel_stats = {}

    for col, corpus in enumerate(["enterprise", "scifact"], start=1):
        div = list(dp[corpus]["divergences"])
        all_y = []
        fits = {}
        for fmt in formats:
            gains = list(dp[corpus]["gains_vs_L1"][fmt])
            slope, intercept, r2, p = ols_fit(div, gains)
            fits[fmt] = (slope, intercept, r2, p, gains)
            all_y.extend(gains)

        # Independent y-range per panel: clip outliers to 2.5–97.5 percentile,
        # symmetric around 0 with padding so regression lines remain visible.
        lo = percentile(all_y, 0.025)
        hi = percentile(all_y, 0.975)
        bound = max(abs(lo), abs(hi))
        bound = max(bound, 2.0) * 1.15
        y_range = [-bound, bound]
        panel_stats[corpus] = {fmt: fits[fmt][:4] for fmt in formats}

        # Faint zero line FIRST so it sits behind data.
        fig.add_shape(
            type="line", x0=0, x1=1, y0=0, y1=0,
            line=dict(color="#bbbbbb", width=1, dash="dot"),
            row=1, col=col,
        )

        for fmt in formats:
            slope, intercept, r2, p, gains = fits[fmt]
            stat = (f"{fmt}  •  β={slope:+.2f},  "
                    f"R²={r2:.03f},  p={p:.03f}")
            # Scatter: hidden from legend (line carries the legend entry),
            # small markers with transparency to expose density.
            fig.add_trace(
                go.Scatter(
                    x=div, y=gains, mode="markers",
                    marker=dict(
                        color=colors[fmt], size=6, opacity=0.40,
                        line=dict(width=0),
                    ),
                    name=fmt, legendgroup=fmt, showlegend=False,
                    hovertemplate=(
                        f"<b>{fmt}</b><br>div=%{{x:.3f}}<br>"
                        "gain=%{y:.0f} ranks<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=xs_line,
                    y=[slope * t + intercept for t in xs_line],
                    mode="lines",
                    line=dict(color=colors[fmt], width=2.6),
                    name=stat, legendgroup=fmt,
                    showlegend=(col == 1),
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )

        fig.update_xaxes(
            title=dict(text="Jaccard divergence", font=dict(size=16)),
            range=[-0.02, 1.02],
            showline=True, linecolor="#888", linewidth=1, mirror=False,
            gridcolor="#ececec", gridwidth=1,
            zeroline=False,
            ticks="outside", tickcolor="#888",
            tickfont=dict(size=14),
            row=1, col=col,
        )
        fig.update_yaxes(
            range=y_range,
            showline=True, linecolor="#888", linewidth=1,
            gridcolor="#ececec", gridwidth=1,
            zeroline=False,
            ticks="outside", tickcolor="#888",
            tickfont=dict(size=14),
            row=1, col=col,
        )

    fig.update_yaxes(
        title=dict(
            text="Per-query rank improvement vs. L1 (ranks)",
            font=dict(size=16),
        ),
        row=1, col=1,
    )

    fig.update_layout(
        font=dict(family=FONT, size=14, color="#222"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text="<b>Vocabulary divergence predicts where augmentation helps</b>",
            x=0.02, xanchor="left", y=0.97, yanchor="top",
            font=dict(size=19, color="#111"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14, family=FONT),
            tracegroupgap=8,
        ),
        margin=dict(l=110, r=40, t=110, b=170),
    )

    write_png(fig, "divergence_law.png", width=1400, height=720)


if __name__ == "__main__":
    fig_reranker_bars()
    fig_per_category()
    fig_divergence_law()
