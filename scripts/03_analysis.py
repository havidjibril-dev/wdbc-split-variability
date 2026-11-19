"""Statistics, tables and figures for the WDBC re-evaluation.

Nested CV: mean of each metric over the 50 outer test folds with a 95% interval from the
corrected resampled t statistic (Nadeau and Bengio 2003), variance factor (1/J + n_test/n_train).
Pairwise comparisons use the same statistic on paired fold differences, Holm-adjusted over
the 6 model pairs within each metric. Primary metrics (fixed before looking at results):
ROC AUC and balanced accuracy at threshold 0.5; everything else is secondary.
"""
import sys, json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from models import MODEL_ORDER

COLORS = {"LR": "#2a78d6", "KNN": "#eb6834", "RF": "#1baf7a", "SVM": "#eda100"}  # validated, light mode
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
NAMES = {"LR": "Logistic regression", "KNN": "k-nearest neighbours", "RF": "Random forest", "SVM": "SVM (RBF)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "ps.fonttype": 42, "pdf.fonttype": 42, "savefig.dpi": 600,
})


def corrected_ci(values, ratio, alpha=0.05):
    """Mean, half-width and p (H0: mean = 0) with the Nadeau-Bengio corrected variance."""
    v = np.asarray(values, float)
    J = len(v)
    mean, var = v.mean(), v.var(ddof=1)
    se = np.sqrt((1 / J + ratio) * var)
    tcrit = stats.t.ppf(1 - alpha / 2, J - 1)
    if se == 0:
        return mean, 0.0, (1.0 if mean == 0 else 0.0)
    p = 2 * stats.t.sf(abs(mean / se), J - 1)
    return mean, tcrit * se, p


def save(fig, stem):
    for ext in ("eps", "pdf", "png"):
        fig.savefig(f"figures/{stem}.{ext}", bbox_inches="tight", dpi=600 if ext != "png" else 200)
    plt.close(fig)


# ---------------------------------------------------------------- nested CV
cv = pd.read_csv("results/nested_cv_folds.csv")
ratio = float((cv["n_test"] / cv["n_train"]).mean())
J = cv["fold"].nunique()

summary_metrics = ["auc", "bal_acc@0.5", "acc@0.5", "sens@0.5", "spec@0.5", "f1@0.5", "mcc@0.5", "brier",
                   "sens@0.95", "spec@0.95", "sens@0.98", "spec@0.98"]
rows = []
for m in MODEL_ORDER:
    sub = cv[cv["model"] == m].sort_values("fold")
    r = {"model": m}
    for k in summary_metrics:
        mean, hw, _ = corrected_ci(sub[k], ratio)
        r[k] = mean; r[k + "_hw"] = hw
    # each sample is tested once per repeat, so totals / repeats = errors per full pass over the data
    reps = sub["repeat"].nunique()
    r["fn_per_pass@0.5"] = sub["fn@0.5"].sum() / reps
    r["fp_per_pass@0.5"] = sub["fp@0.5"].sum() / reps
    r["fn_per_pass@0.95"] = sub["fn@0.95"].sum() / reps
    r["fp_per_pass@0.95"] = sub["fp@0.95"].sum() / reps
    r["fit_ms_median"] = 1000 * sub["fit_s"].median()
    r["pred_ms_median"] = 1000 * sub["pred_s"].median()
    rows.append(r)
summary = pd.DataFrame(rows)
summary.to_csv("results/summary_nested.csv", index=False)

pair_metrics = ["auc", "bal_acc@0.5", "acc@0.5", "sens@0.5", "spec@0.5", "brier", "spec@0.95"]
prows = []
wide = {k: cv.pivot(index="fold", columns="model", values=k) for k in pair_metrics}
for k in pair_metrics:
    block = []
    for a, b in combinations(MODEL_ORDER, 2):
        d = wide[k][a] - wide[k][b]
        mean, hw, p = corrected_ci(d, ratio)
        block.append({"metric": k, "pair": f"{a} - {b}", "a": a, "b": b, "diff": mean, "lo": mean - hw,
                      "hi": mean + hw, "p": p, "a_better_folds": int((d > 0).sum()), "ties": int((d == 0).sum()),
                      "b_better_folds": int((d < 0).sum())})
    padj = multipletests([x["p"] for x in block], method="holm")[1]
    for x, q in zip(block, padj):
        x["p_holm"] = q
        x["primary"] = k in ("auc", "bal_acc@0.5")
    prows += block
pairwise = pd.DataFrame(prows)
pairwise.to_csv("results/pairwise_nested.csv", index=False)

# chosen hyperparameters
prm = cv.groupby(["model", "params"]).size().rename("folds").reset_index()
prm = prm.sort_values(["model", "folds"], ascending=[True, False])
prm.to_csv("results/params_freq_nested.csv", index=False)

# ---------------------------------------------------------------- single split
ss = pd.read_csv("results/single_split.csv")
acc = ss.pivot(index="seed", columns="model", values="test_acc")[MODEL_ORDER]
best = acc.max(axis=1)
winners = acc.eq(best, axis=0)
share = winners.div(winners.sum(axis=1), axis=0).sum() / len(acc)  # ties split equally
sole = (winners.sum(axis=1) == 1)
srows = []
for m in MODEL_ORDER:
    a = acc[m]
    srows.append({"model": m, "mean": a.mean(), "sd": a.std(ddof=1), "min": a.min(), "q05": a.quantile(0.05),
                  "median": a.median(), "q95": a.quantile(0.95), "max": a.max(), "seed42": a.loc[42],
                  "win_share": share[m], "sole_wins": int((winners[m] & sole).sum()),
                  "tied_for_best": int(winners[m].sum())})
ssum = pd.DataFrame(srows)
ssum.to_csv("results/summary_single_split.csv", index=False)
extra = {
    "n_splits": len(acc),
    "splits_with_tie_for_best": int((~sole).sum()),
    "svm_minus_rf_acc_pos_zero_neg": [int((acc.SVM > acc.RF).sum()), int((acc.SVM == acc.RF).sum()), int((acc.SVM < acc.RF).sum())],
    "svm_minus_lr_acc_pos_zero_neg": [int((acc.SVM > acc.LR).sum()), int((acc.SVM == acc.LR).sum()), int((acc.SVM < acc.LR).sum())],
    "share_svm_at_least_0.9824": float((acc.SVM >= 112 / 114 - 1e-9).mean()),
    "seed42_acc": acc.loc[42].to_dict(),
    "seed42_params": ss[ss.seed == 42].set_index("model")["params"].to_dict(),
}
with open("results/single_split_extra.json", "w") as f:
    json.dump(extra, f, indent=2)

# ---------------------------------------------------------------- Figure 1: single-split spread
rng = np.random.default_rng(0)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.45})
for i, m in enumerate(MODEL_ORDER):
    y = 100 * acc[m].values
    x = i + rng.uniform(-0.22, 0.22, len(y))
    ax.scatter(x, y, s=10, color=COLORS[m], edgecolors="white", linewidths=0.3, zorder=2)
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    ax.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=1.2, zorder=3)
    ax.plot([i, i], [q1, q3], color=INK, lw=0.8, zorder=3)
    ax.scatter([i + 0.38], [100 * acc.loc[42, m]], marker="D", s=22, color=INK, zorder=4)
ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.3)
ax.set_ylabel("Test accuracy (%)")
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.scatter([], [], marker="D", s=22, color=INK, label="Split used in the course report (seed 42)")
ax.plot([], [], color=INK, lw=1.2, label="Median and interquartile range")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, handletextpad=0.4,
          columnspacing=1.2, borderaxespad=0)
ax.text(-0.18, 1.14, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)

order = list(MODEL_ORDER)
bx.barh(range(len(order)), [100 * share[m] for m in order], color=[COLORS[m] for m in order], height=0.6,
        edgecolor="white", linewidth=1)
for i, m in enumerate(order):
    bx.text(100 * share[m] + 1, i, f"{100 * share[m]:.0f}%", va="center", fontsize=8, color=INK)
bx.set_yticks(range(len(order)), order); bx.invert_yaxis()
bx.set_xlabel("Share of splits won (%)")
bx.set_xlim(0, max(100 * share.max() * 1.3, 10))
bx.xaxis.grid(True, color=GRID, lw=0.5); bx.set_axisbelow(True)
bx.text(-0.32, 1.14, "(b)", transform=bx.transAxes, fontsize=9, fontweight="bold", color=INK)
save(fig, "Fig1")

# ---------------------------------------------------------------- Figure 2: paired differences
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
labels = [f"{a} − {b}" for a, b in combinations(MODEL_ORDER, 2)]
for ax, k, xl, scale in zip(axes, ["auc", "bal_acc@0.5"],
                            ["Difference in ROC AUC", "Difference in balanced accuracy (pp)"], [1, 100]):
    blk = pairwise[pairwise.metric == k].reset_index(drop=True)
    for i, r in blk.iterrows():
        sig = r.p_holm < 0.05
        ax.plot([scale * r.lo, scale * r.hi], [i, i], color=INK if sig else MUTED, lw=1.2)
        ax.scatter([scale * r["diff"]], [i], s=24, color=INK if sig else "white", edgecolors=INK if sig else MUTED,
                   linewidths=1, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(xl)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
axes[0].invert_yaxis()  # shared y axis: invert once so the first pair is on top
axes[0].text(-0.36, 1.14, "(a)", transform=axes[0].transAxes, fontsize=9, fontweight="bold", color=INK)
axes[1].text(-0.04, 1.14, "(b)", transform=axes[1].transAxes, fontsize=9, fontweight="bold", color=INK)
axes[0].scatter([], [], s=24, color=INK, label="Holm-adjusted p < 0.05")
axes[0].scatter([], [], s=24, color="white", edgecolors=MUTED, label="Not significant after Holm adjustment")
axes[0].plot([], [], color=MUTED, lw=1.2, label="Unadjusted 95% interval (corrected resampled t)")
axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0)
save(fig, "Fig2")

pd.set_option("display.width", 200)
print(summary[["model", "auc", "auc_hw", "bal_acc@0.5", "bal_acc@0.5_hw", "acc@0.5", "sens@0.5", "spec@0.5",
               "brier", "sens@0.95", "spec@0.95", "fn_per_pass@0.5", "fp_per_pass@0.5", "fit_ms_median"]].round(4))
print(pairwise[pairwise.primary][["metric", "pair", "diff", "lo", "hi", "p", "p_holm"]].round(4))
print(ssum.round(4))
print(json.dumps(extra, indent=1))
