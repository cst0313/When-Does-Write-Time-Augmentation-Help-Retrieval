"""
Robustness analyses to address statistical critique:

1. Multiple-comparisons corrections (Holm-Bonferroni, BH-FDR) on the
   paired-bootstrap p-values in Table 3.
2. Cohen's kappa + binomial test on the 16/20 diagnostic calibration.
3. Split-half cross-validation of the diagnostic on enterprise:
   estimate β on 80 queries, test on the other 80; report agreement
   between in-sample and out-of-sample sign + significance.
4. ROC sweep of the diagnostic's p-value threshold.
5. Power analysis: minimum detectable effect size at n=80 with
   paired-bootstrap CI half-width = 0.05.
"""

import json
import math
import random
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
random.seed(0)


# ---------- (1) Multiple-comparisons corrections on Table 3 ----------

# Table 3 reports 18 paired-bootstrap p-values:
TABLE3 = [
    # name, p, family
    ("RODS-M6 vs L1, in-dist R@5",       0.040, "R@5_indist"),
    ("RODS-M4 vs L1, in-dist R@5",       0.038, "R@5_indist"),
    ("docT5query vs L1, in-dist R@5",    0.036, "R@5_indist"),
    ("RODS-M6 vs docT5query, in-dist R@5", 0.52, "R@5_indist"),
    ("Three-vec vs L1, in-dist R@5",     0.013, "R@5_indist"),
    ("RODS-M6 vs L1, held-out R@5",      0.73,  "R@5_held"),
    ("RODS-M4 vs L1, held-out R@5",      0.72,  "R@5_held"),
    ("docT5query vs L1, held-out R@5",   0.88,  "R@5_held"),
    ("RODS-M6 vs docT5query, held-out R@5", 0.74, "R@5_held"),
    ("Three-vec vs L1, held-out R@5",    0.10,  "R@5_held"),
    ("RODS-M6 vs L1, in-dist MRR",       0.10,  "MRR_indist"),
    ("RODS-M4 vs L1, in-dist MRR",       0.047, "MRR_indist"),
    ("docT5query vs L1, in-dist MRR",    0.003, "MRR_indist"),
    ("Three-vec vs L1, in-dist MRR",     0.030, "MRR_indist"),
    ("RODS-M6 vs L1, held-out MRR",      0.41,  "MRR_held"),
    ("RODS-M4 vs L1, held-out MRR",      0.011, "MRR_held"),
    ("docT5query vs L1, held-out MRR",   0.88,  "MRR_held"),
    ("Three-vec vs L1, held-out MRR",    0.001, "MRR_held"),
]


def holm_bonferroni(ps, alpha=0.05):
    """Holm-Bonferroni step-down. Returns list of (idx, p_adj, reject)."""
    n = len(ps)
    indexed = sorted(enumerate(ps), key=lambda x: x[1])
    out = [None] * n
    for rank, (idx, p) in enumerate(indexed):
        p_adj = min(1.0, p * (n - rank))
        # Holm: reject if all earlier (smaller) p_adj also rejected
        reject = p_adj < alpha
        # enforce monotonicity
        if rank > 0 and out[indexed[rank - 1][0]][1] > p_adj:
            p_adj = out[indexed[rank - 1][0]][1]
        out[idx] = (idx, p_adj, reject)
    # Now enforce step-down: if any earlier failed to reject, all later must fail
    fail_seen = False
    for rank, (idx, _) in enumerate(indexed):
        _, p_adj, reject = out[idx]
        if fail_seen:
            reject = False
        elif not reject:
            fail_seen = True
        out[idx] = (idx, p_adj, reject)
    return out


def bh_fdr(ps, alpha=0.05):
    """Benjamini-Hochberg step-up. Returns list of (idx, p_adj, reject)."""
    n = len(ps)
    indexed = sorted(enumerate(ps), key=lambda x: x[1])
    out = [None] * n
    # BH adjusted p-values via cumulative min from the back
    adj = [0.0] * n
    cur_min = 1.0
    for rank in range(n - 1, -1, -1):
        idx, p = indexed[rank]
        adj_p = p * n / (rank + 1)
        cur_min = min(cur_min, adj_p)
        adj[idx] = min(1.0, cur_min)
    for idx, p in enumerate(ps):
        out[idx] = (idx, adj[idx], adj[idx] < alpha)
    return out


def multiple_comparisons():
    print("=" * 80)
    print("MULTIPLE COMPARISONS CORRECTIONS ON TABLE 3 P-VALUES (n=18)")
    print("=" * 80)
    ps = [t[1] for t in TABLE3]
    holm = holm_bonferroni(ps)
    bh = bh_fdr(ps)
    print(f"  {'Comparison':<42} {'p':>7} {'Holm':>8}  rej   {'BH':>8}  rej")
    print(f"  {'-'*42} {'-'*7} {'-'*8} ---  {'-'*8} ---")
    for (name, p, _), (_, h_p, h_r), (_, b_p, b_r) in zip(TABLE3, holm, bh):
        print(f"  {name:<42} {p:>7.3f} {h_p:>8.3f}   {'X' if h_r else ' '}   {b_p:>8.3f}   {'X' if b_r else ' '}")
    n_orig = sum(1 for t in TABLE3 if t[1] < 0.05)
    n_holm = sum(1 for x in holm if x[2])
    n_bh = sum(1 for x in bh if x[2])
    print(f"\n  Significant @ alpha=0.05:  uncorrected={n_orig}  Holm-Bonferroni={n_holm}  BH-FDR={n_bh}")
    return holm, bh


# ---------- (2) Cohen's kappa + binomial on calibration ----------

def kappa_calibration():
    print()
    print("=" * 80)
    print("DIAGNOSTIC CALIBRATION: COHEN'S KAPPA + BINOMIAL TEST")
    print("=" * 80)
    # 20 cells from Table 11
    # (pred, obs)
    cells = [
        # Enterprise
        (1, 1), (1, 1), (0, 1), (1, 1),
        # SciFact
        (0, 0), (0, 0), (0, 0), (0, 0),
        # NFCorpus
        (0, 1), (0, 1), (0, 1), (0, 0),
        # FiQA
        (0, 0), (0, 0), (0, 0), (0, 0),
        # LoTTE
        (0, 0), (0, 0), (0, 0), (0, 0),
    ]
    n = len(cells)
    n_agree = sum(1 for p, o in cells if p == o)
    pred_pos = sum(1 for p, o in cells if p == 1)
    obs_pos = sum(1 for p, o in cells if o == 1)
    p_obs_marginal = obs_pos / n  # prob "obs lift"
    p_pred_marginal = pred_pos / n
    p_agree_chance = (p_pred_marginal * p_obs_marginal +
                      (1 - p_pred_marginal) * (1 - p_obs_marginal))
    p_agree_obs = n_agree / n
    kappa = (p_agree_obs - p_agree_chance) / (1 - p_agree_chance)
    print(f"  n cells = {n}")
    print(f"  pred lift = {pred_pos}/{n}, obs lift = {obs_pos}/{n}")
    print(f"  observed agreement = {n_agree}/{n} = {p_agree_obs:.3f}")
    print(f"  chance agreement = {p_agree_chance:.3f}")
    print(f"  Cohen's kappa = {kappa:.3f}  (Landis-Koch: moderate agreement)")
    # Binomial test against p=p_agree_chance
    from math import comb
    p_value = sum(comb(n, k) * p_agree_chance**k * (1 - p_agree_chance)**(n - k)
                  for k in range(n_agree, n + 1))
    print(f"  Binomial p (X >= 16 | n=20, p_chance={p_agree_chance:.3f}): {p_value:.4f}")
    # Versus naive baseline (always say "no lift" gets 13/20)
    naive_correct = n - obs_pos
    print(f"  Always-say-no-lift baseline would agree on {naive_correct}/{n}")
    print(f"  Diagnostic improves over naive by {n_agree - naive_correct} cells")
    return kappa, p_value


# ---------- (3) Split-half cross-validation ----------

def linreg(x, y):
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    mx = sum(x) / n; my = sum(y) / n
    sxx = sum((xi - mx)**2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if sxx == 0:
        return 0.0, my
    return sxy / sxx, my - (sxy / sxx) * mx


def perm_p_slope(x, y, n_perm=2000):
    obs, _ = linreg(x, y)
    cnt = 0
    y_pool = list(y)
    for _ in range(n_perm):
        random.shuffle(y_pool)
        b, _ = linreg(x, y_pool)
        if abs(b) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def split_half_cv():
    print()
    print("=" * 80)
    print("SPLIT-HALF CROSS-VALIDATION OF THE DIAGNOSTIC ON ENTERPRISE")
    print("=" * 80)
    with open(DATA / "divergence_predictor.json") as f:
        d = json.load(f)
    div = d["enterprise"]["divergences"]
    n = len(div)
    print(f"  n total = {n}, splitting in half repeatedly")
    formats = ["RODS-M4", "RODS-M6", "docT5query", "Three-vector"]
    n_iter = 100  # 100 random split-halves
    for fmt in formats:
        gain = d["enterprise"]["gains_vs_L1"][fmt]
        beta_full, p_full = perm_p_slope(div, gain, n_perm=2000)
        # Random split-halves
        sign_consistent = 0
        sig_consistent = 0
        for _ in range(n_iter):
            idx = list(range(n))
            random.shuffle(idx)
            half_a = idx[: n // 2]
            half_b = idx[n // 2 :]
            x_a = [div[i] for i in half_a]; y_a = [gain[i] for i in half_a]
            x_b = [div[i] for i in half_b]; y_b = [gain[i] for i in half_b]
            ba, pa = perm_p_slope(x_a, y_a, n_perm=300)
            bb, pb = perm_p_slope(x_b, y_b, n_perm=300)
            # Sign-consistent: both halves agree on sign of slope
            if (ba > 0 and bb > 0) or (ba < 0 and bb < 0):
                sign_consistent += 1
            # Sig-consistent: both halves agree on sign AND both p<0.10
            if (ba > 0 and bb > 0 and pa < 0.10 and pb < 0.10) or \
               (ba < 0 and bb < 0 and pa < 0.10 and pb < 0.10):
                sig_consistent += 1
        print(f"  {fmt:<14} full β={beta_full:+.2f} p={p_full:.3f}  | " +
              f"split-half sign agree {sign_consistent}/{n_iter}, " +
              f"both p<0.10 {sig_consistent}/{n_iter}")


# ---------- (4) Power / minimum detectable effect ----------

def power_analysis():
    print()
    print("=" * 80)
    print("POWER ANALYSIS")
    print("=" * 80)
    # Approximate normal power calc for paired comparison
    # n=80, alpha=0.05 two-sided, power=0.80
    # For paired-bootstrap on R@5 with observed CI half-width sigma:
    # MDE = (z_{alpha/2} + z_{1-beta}) * sigma_diff / sqrt(n)
    # where sigma_diff is the SD of paired differences
    # We approximate sigma_diff ~= 0.20 (matches observed CI half-widths)
    z_alpha = 1.96
    z_power = 0.84
    for n in [50, 80, 160, 300, 1000]:
        for sigma_diff in [0.20, 0.30]:
            mde = (z_alpha + z_power) * sigma_diff / math.sqrt(n)
            print(f"  n={n:<4} sigma_diff={sigma_diff:.2f}   MDE @80% power = {mde:.3f}")
    print()
    print("  Interpretation: with n=80 and observed paired-difference SD ~=0.20,")
    print("  minimum detectable R@5 lift at 80% power is ~6.2 pp.")
    print("  Reported in-distribution lift of +3.7 pp is at ~50% power (under-powered).")


# ---------- (5) ROC sweep of diagnostic threshold ----------

def roc_sweep():
    print()
    print("=" * 80)
    print("ROC SWEEP OF DIAGNOSTIC THRESHOLD (alpha varies)")
    print("=" * 80)
    with open(DATA / "diagnostic_validation.json") as f:
        rows = json.load(f)
    print(f"  {'alpha':>7}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  {'TPR':>5} {'FPR':>5} {'kappa':>6}")
    for alpha in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        tp = fp = fn = tn = 0
        for r in rows:
            pred = (r["beta"] > 0) and (r["p"] < alpha)
            obs = r["observed_mean_gain"] > 0.05
            if pred and obs: tp += 1
            elif pred and not obs: fp += 1
            elif not pred and obs: fn += 1
            else: tn += 1
        n = tp + fp + fn + tn
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        # kappa
        po = (tp + tn) / n
        pe = ((tp + fp)/n) * ((tp + fn)/n) + ((fn + tn)/n) * ((fp + tn)/n)
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
        print(f"  {alpha:>7.2f}  {tp:>3} {fp:>3} {fn:>3} {tn:>3}  {tpr:>5.2f} {fpr:>5.2f} {kappa:>6.3f}")


def main():
    multiple_comparisons()
    kappa_calibration()
    split_half_cv()
    power_analysis()
    roc_sweep()


if __name__ == "__main__":
    main()
