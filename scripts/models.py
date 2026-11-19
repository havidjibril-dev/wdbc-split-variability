"""Model definitions and hyperparameter grids shared by all experiments.

Every model that depends on feature scale has the scaler inside its pipeline, so the
scaler is fitted only on the data the model is trained on (no leakage into CV folds).
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

MODEL_ORDER = ["LR", "KNN", "RF", "SVM"]


def make_models(seed):
    """Return {name: (estimator, param_grid)} with a fold-specific seed for RF and SVM."""
    return {
        "LR": (
            Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=10000))]),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        ),
        "KNN": (
            Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {
                "clf__n_neighbors": list(range(1, 72, 2)),
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        ),
        "RF": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            {"max_features": ["sqrt", "log2", 0.5], "min_samples_leaf": [1, 3, 5]},
        ),
        "SVM": (
            Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=seed))]),
            {"clf__C": [0.1, 1, 10, 100, 1000], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
        ),
    }


def with_probabilities(est):
    """Give SVC Platt-scaled probabilities.

    SVC(probability=True) is deprecated in scikit-learn 1.9; the documented replacement is
    CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False), which fits the sigmoid on
    internal 5-fold decision values and the SVC on all training rows.
    """
    if isinstance(est, Pipeline) and isinstance(est.named_steps["clf"], SVC):
        svc = est.named_steps["clf"]
        est.steps[-1] = ("clf", CalibratedClassifierCV(svc, method="sigmoid", ensemble=False, cv=5))
    return est
