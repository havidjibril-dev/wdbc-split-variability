"""Sanity check: reproduce the original course report's single 80/20 split."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from common import load_wdbc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

X, y, _ = load_wdbc()
print("n =", len(y), "malignant =", y.sum(), "benign =", (y == 0).sum())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
ss = StandardScaler().fit(Xtr); Xtr, Xte = ss.transform(Xtr), ss.transform(Xte)

knn = KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="euclidean").fit(Xtr, ytr)
p = knn.predict(Xte); print("KNN k=3 acc", accuracy_score(yte, p), confusion_matrix(yte, p).ravel())

svc = SVC(kernel="rbf").fit(Xtr, ytr)
p = svc.predict(Xte); print("SVM default acc", accuracy_score(yte, p), confusion_matrix(yte, p).ravel())

grid = GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001], "kernel": ["rbf", "poly"]}, cv=5).fit(Xtr, ytr)
p = grid.predict(Xte)
print("SVM tuned", grid.best_params_, "cv", round(grid.best_score_, 4), "acc", accuracy_score(yte, p), "tn fp fn tp", confusion_matrix(yte, p).ravel(),
      "AUC", round(roc_auc_score(yte, grid.decision_function(Xte)), 4))

accs = []
for s in range(20):
    rf = RandomForestClassifier(n_estimators=300, random_state=s).fit(Xtr, ytr)
    accs.append(accuracy_score(yte, rf.predict(Xte)))
print("RF 300 trees over 20 seeds: test acc min/median/max", min(accs), np.median(accs), max(accs))
