"""Shared data loading for the WDBC re-evaluation study."""
import numpy as np
from sklearn.datasets import load_breast_cancer


def load_wdbc():
    """Return X (569x30), y with malignant=1, benign=0, and feature names.

    sklearn encodes 0=malignant, 1=benign; we flip so the positive class is malignant,
    matching the original report (B->0, M->1).
    """
    d = load_breast_cancer()
    X = d.data.astype(float)
    y = (1 - d.target).astype(int)
    return X, y, list(d.feature_names)
