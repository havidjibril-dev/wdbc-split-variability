"""How stable is a single 80/20 hold-out comparison?

Repeats the common single-split protocol 100 times (split seeds 0..99, which include
seed 42 used in the original report): stratified 80/20 split, 5-fold grid search on
accuracy on the 80%, refit, accuracy on the 20%. The same grids as the nested CV are used.
Output: results/single_split.csv (one row per seed x model).
"""
import sys, json, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

sys.path.insert(0, "scripts")
from common import load_wdbc
from models import make_models, MODEL_ORDER

SEEDS = range(100)


def run_seed(seed, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, (est, grid) in make_models(seed=seed).items():
        gs = GridSearchCV(est, grid, scoring="accuracy", cv=cv, n_jobs=1).fit(Xtr, ytr)
        yhat = gs.predict(Xte)
        score = gs.decision_function(Xte) if hasattr(gs.best_estimator_, "decision_function") else gs.predict_proba(Xte)[:, 1]
        tn, fp, fn, tp = confusion_matrix(yte, yhat, labels=[0, 1]).ravel()
        rows.append({"seed": seed, "model": name, "params": json.dumps(gs.best_params_, default=str),
                     "cv_acc": gs.best_score_, "test_acc": accuracy_score(yte, yhat),
                     "test_auc": roc_auc_score(yte, score), "fn": int(fn), "fp": int(fp), "n_test": len(yte)})
    return rows


if __name__ == "__main__":
    X, y, _ = load_wdbc()
    t0 = time.time()
    out = Parallel(n_jobs=2, verbose=5)(delayed(run_seed)(s, X, y) for s in SEEDS)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/single_split.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s, rows = {len(df)}")
    print(df.groupby("model")["test_acc"].describe().loc[MODEL_ORDER].round(4))
