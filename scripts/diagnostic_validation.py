"""
Diagnostic-loop validation + 50-query subsample bootstrap.

Closes two reviewer-flagged gaps:

1. The diagnostic (regress per-query rank gain on Jaccard divergence)
   is described but never demonstrated as a *prediction*. We compute
   for every corpus we evaluate (enterprise, scifact, nfcorpus, fiqa,
   lottewriting5k) the diagnostic's prediction (β, p, mean div) AND
   the observed augmentation lift, and report them in a 5-corpus
   table.

2. The "~50 held-out pairs" recipe is unverified. We subsample 50
   queries from the enterprise n=160 set 1000 times and report the
   distribution of β / p that the diagnostic recovers on each
   subsample.

Outputs:
    data/diagnostic_validation.json
    data/diagnostic_subsample_50.json
"""

import json
import math
import random
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
DIV_PATH = DATA / "divergence_predictor.json"
OUT_VAL = DATA / "diagnostic_validation.json"
OUT_SUB = DATA / "diagnostic_subsample_50.json"

CORPORA = ["enterprise", "scifact", "nfcorpus", "fiqa", "lottewriting5k"]
RODS_FORMATS = ["RODS-M4", "RODS-M6", "docT5query", "Three-vector"]
PRIMARY = "RODS-M6"  # the regression headline
random.seed(0)


def linreg(x, y):
    """Simple OLS slope/intercept/R^2."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0:
        return 0.0, my, 0.0
    beta = sxy / sxx
    alpha = my - beta * mx
    if syy == 0:
        r2 = 0.0
    else:
        r2 = (sxy ** 2) / (sxx * syy)
    return beta, alpha, r2


def perm_p_slope(x, y, n_perm=2000):
    """Permutation p for slope, two-sided."""
    obs, _, _ = linreg(x, y)
    cnt = 0
    y_pool = list(y)
    for _ in range(n_perm):
        random.shuffle(y_pool)
        b, _, _ = linreg(x, y_pool)
        if abs(b) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def diagnostic_for(div, gain):
    """Run diagnostic on (div, gain) lists; return summary."""
    mean_div = sum(div) / len(div)
    beta, alpha, r2 = linreg(div, gain)
    p = perm_p_slope(div, gain, n_perm=2000)
    observed_mean_gain = sum(gain) / len(gain)
    return {
        "n": len(div),
        "mean_div": mean_div,
        "beta": beta,
        "alpha": alpha,
        "r2": r2,
        "p": p,
        "observed_mean_gain": observed_mean_gain,
    }


def diagnostic_predicts_lift(summary, beta_threshold=0.0, p_threshold=0.10):
    """Return True if diagnostic predicts a measurable lift."""
    return (summary["beta"] > beta_threshold) and (summary["p"] < p_threshold)


def observed_lift_present(summary, gain_threshold=0.05):
    """Observed lift = positive mean rank gain above noise threshold."""
    return summary["observed_mean_gain"] > gain_threshold


def main():
    with open(DIV_PATH) as f:
        d = json.load(f)

    rows = []
    for corpus in CORPORA:
        if corpus not in d:
            continue
        cd = d[corpus]
        div = cd["divergences"]
        for fmt in RODS_FORMATS:
            if fmt not in cd["gains_vs_L1"]:
                continue
            gain = cd["gains_vs_L1"][fmt]
            s = diagnostic_for(div, gain)
            s["corpus"] = corpus
            s["format"] = fmt
            s["predicts_lift"] = diagnostic_predicts_lift(s)
            s["observed_lift"] = observed_lift_present(s)
            s["agrees"] = s["predicts_lift"] == s["observed_lift"]
            rows.append(s)

    # Print summary
    print(f"\n{'Corpus':<18} {'Format':<14} {'div̄':>6} {'β':>7} {'p':>7} {'R²':>6} {'gain':>7}  pred  obs  agree")
    print("─" * 95)
    for r in rows:
        pred = "✓" if r["predicts_lift"] else "·"
        obs = "✓" if r["observed_lift"] else "·"
        agree = "✓" if r["agrees"] else "✗"
        print(f"  {r['corpus']:<16} {r['format']:<14} {r['mean_div']:>6.2f} {r['beta']:>+7.2f} {r['p']:>7.3f} {r['r2']:>6.3f} {r['observed_mean_gain']:>+7.2f}    {pred}    {obs}    {agree}")

    n_total = len(rows)
    n_agree = sum(1 for r in rows if r["agrees"])
    print(f"\n  Diagnostic agrees with observed outcome on {n_agree}/{n_total} (corpus, format) cells.")

    with open(OUT_VAL, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved → {OUT_VAL}")

    # 50-query subsample bootstrap on enterprise (n=160) for RODS-M6
    print(f"\n=== 50-query subsample bootstrap (enterprise, {PRIMARY}) ===")
    div = d["enterprise"]["divergences"]
    gain = d["enterprise"]["gains_vs_L1"][PRIMARY]
    n_full = len(div)
    n_sub = 50
    n_boot = 1000

    betas, ps = [], []
    for b in range(n_boot):
        idx = random.sample(range(n_full), n_sub)
        x = [div[i] for i in idx]
        y = [gain[i] for i in idx]
        beta, _, _ = linreg(x, y)
        # Quick perm test (300 perms each for speed) for sign coverage
        p = perm_p_slope(x, y, n_perm=300)
        betas.append(beta)
        ps.append(p)

    betas.sort()
    ps_sorted = sorted(ps)
    median_beta = betas[n_boot // 2]
    lo_beta = betas[int(n_boot * 0.025)]
    hi_beta = betas[int(n_boot * 0.975)]
    pct_pos = sum(1 for b in betas if b > 0) / n_boot
    pct_pos_sig = sum(1 for b, p in zip(betas, ps) if b > 0 and p < 0.10) / n_boot
    pct_pos_sig_05 = sum(1 for b, p in zip(betas, ps) if b > 0 and p < 0.05) / n_boot

    print(f"  Full sample (n={n_full}): β = (computed elsewhere)")
    print(f"  Subsample n=50, B={n_boot}:")
    print(f"    β median = {median_beta:+.3f}  [95% CI {lo_beta:+.3f}, {hi_beta:+.3f}]")
    print(f"    Pr(β > 0)            = {pct_pos:.3f}")
    print(f"    Pr(β > 0 and p<0.10) = {pct_pos_sig:.3f}")
    print(f"    Pr(β > 0 and p<0.05) = {pct_pos_sig_05:.3f}")

    out = {
        "format": PRIMARY,
        "corpus": "enterprise",
        "n_full": n_full,
        "n_sub": n_sub,
        "n_boot": n_boot,
        "betas": betas,
        "ps": ps,
        "median_beta": median_beta,
        "ci95_beta": [lo_beta, hi_beta],
        "pct_pos": pct_pos,
        "pct_pos_p10": pct_pos_sig,
        "pct_pos_p05": pct_pos_sig_05,
    }
    with open(OUT_SUB, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_SUB}")


if __name__ == "__main__":
    main()
