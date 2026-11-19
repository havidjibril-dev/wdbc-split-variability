"""Repeated stratified nested cross-validation of LR, KNN, RF and SVM on WDBC.

Outer loop: 10 repeats x 5 folds (50 test folds), seed 2026.
Inner loop: 5-fold stratified grid search on ROC AUC, then
  * the best configuration is refitted on the outer training part,
  * inner out-of-fold probabilities of the best configuration give the decision
    thresholds that reach a target sensitivity (0.95 and 0.98) on training data only,
  * metrics are computed on the untouched outer test fold.
Output: results/nested_cv_folds.csv (one row per fold x model).
"""
import sys, json, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss, accuracy_score, balanced_accuracy_score,
                             recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix)

sys.path.insert(0, "scripts")
from common import load_wdbc
from models import make_models, with_probabilities, MODEL_ORDER

N_SPLITS, N_REPEATS, OUTER_SEED = 5, 10, 2026
TARGET_SENS = (0.95, 0.98)


def threshold_for_sensitivity(y, p, target):
    """Largest threshold t such that the share of positives with p >= t is at least target."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos)))
    return float(pos[k - 1])


def metrics_at(y, p, t):
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y, yhat),
        "bal_acc": balanced_accuracy_score(y, yhat),
        "sens": recall_score(y, yhat, zero_division=0),
        "spec": tn / (tn + fp),
        "prec": precision_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_fold(fold_id, tr, te, X, y):
    rows = []
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + fold_id)
    for name, (est, grid) in make_models(seed=fold_id).items():
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=False)
        gs.fit(X[tr], y[tr])
        best = with_probabilities(clone(est).set_params(**gs.best_params_))

        # thresholds from inner out-of-fold probabilities (training data only)
        p_oof = cross_val_predict(clone(best), X[tr], y[tr], cv=inner, method="predict_proba", n_jobs=1)[:, 1]
        thr = {s: threshold_for_sensitivity(y[tr], p_oof, s) for s in TARGET_SENS}

        t0 = time.perf_counter(); best.fit(X[tr], y[tr]); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = best.predict_proba(X[te])[:, 1]; pred_s = time.perf_counter() - t0

        row = {"fold": fold_id, "repeat": fold_id // N_SPLITS, "model": name,
               "params": json.dumps(gs.best_params_, default=str), "inner_auc": gs.best_score_,
               "auc": roc_auc_score(y[te], p), "brier": brier_score_loss(y[te], p),
               "logloss": log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6)),
               "fit_s": fit_s, "pred_s": pred_s, "n_train": len(tr), "n_test": len(te)}
        row.update({f"{k}@0.5": v for k, v in metrics_at(y[te], p, 0.5).items()})
        for s in TARGET_SENS:
            row[f"thr@{s}"] = thr[s]
            row.update({f"{k}@{s}": v for k, v in metrics_at(y[te], p, thr[s]).items()})
        rows.append(row)
    return rows


if __name__ == "__main__":
    X, y, _ = load_wdbc()
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=OUTER_SEED)
    t_start = time.time()
    out = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(i, tr, te, X, y) for i, (tr, te) in enumerate(outer.split(X, y)))
    df = pd.DataFrame([r for rows in out for r in rows])
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df.sort_values(["fold", "model"]).to_csv("results/nested_cv_folds.csv", index=False)
    print(f"done in {time.time() - t_start:.0f} s, rows = {len(df)}")
    print(df.groupby("model", observed=True)[["auc", "acc@0.5", "bal_acc@0.5", "sens@0.5", "spec@0.5", "brier",
                                               "sens@0.95", "spec@0.95"]].mean().round(4))
