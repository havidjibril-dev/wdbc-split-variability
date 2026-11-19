# Split variability and classifier rankings on the Wisconsin Diagnostic Breast Cancer data

Code and results for the manuscript *Split variability and classifier rankings on the Wisconsin
Diagnostic Breast Cancer data: a repeated nested cross-validation study* (H. J. Pamenan, submitted to
SN Computer Science).

The study compares logistic regression, k-nearest neighbors, random forest and an RBF support vector
machine on the WDBC data in two ways:

1. 100 stratified 80/20 hold-out splits (seeds 0 to 99), each with grid search on the training part, to
   measure how much single-split test accuracy and the model ranking change with the split;
2. 10 x 5 repeated stratified nested cross-validation with inner grid search on ROC AUC, thresholds for a
   target sensitivity chosen on training data only, and paired comparisons with the corrected resampled
   t-test (Nadeau and Bengio, 2003) and the Holm adjustment.

## Data

Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 features. W. Wolberg, O. Mangasarian, N. Street and
W. Street (1993), UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B, licensed CC BY 4.0.
The scripts load the copy distributed with scikit-learn (`sklearn.datasets.load_breast_cancer`), so no
download is needed. Malignant is coded as the positive class.

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/00_reproduce_report.py   # the original course-project split (seed 42), < 1 min
python scripts/01_nested_cv.py          # nested cross-validation, about 12 min on 2 CPU cores
python scripts/02_single_split.py       # 100 hold-out splits, about 21 min on 2 CPU cores
python scripts/03_analysis.py           # statistics, summary tables, Fig1 and Fig2, < 1 min
```

Run the scripts from the repository root. `scripts/models.py` defines the pipelines and grids used by all
experiments; scalers sit inside the pipelines, so they are fitted only on training folds.

## Outputs

| File | Content |
|---|---|
| `results/nested_cv_folds.csv` | One row per outer fold and model: chosen hyperparameters, AUC, Brier score, metrics at threshold 0.5 and at thresholds for sensitivity 0.95 and 0.98, timing |
| `results/summary_nested.csv` | Means with 95% half-widths (`*_hw`, corrected resampled t), errors per full pass over the data, median fit and prediction times |
| `results/pairwise_nested.csv` | Paired differences, 95% intervals, p and Holm-adjusted p |
| `results/params_freq_nested.csv` | Hyperparameters chosen per outer fold |
| `results/single_split.csv` | One row per split seed and model |
| `results/summary_single_split.csv`, `results/single_split_extra.json` | Spread of test accuracy, win shares, seed-42 values |
| `figures/Fig1.*`, `figures/Fig2.*` | Figures of the manuscript (EPS, PDF, PNG) |

## Seeds and versions

Outer CV seed 2026; inner CV seed 1000 + outer fold index; random forest and SVM seed = outer fold index;
hold-out seeds 0 to 99. Results were produced with Python 3.13.18 and the package versions in
`requirements.txt`. SVM probabilities use `CalibratedClassifierCV(SVC(...), method="sigmoid",
ensemble=False, cv=5)` because `SVC(probability=True)` is deprecated in scikit-learn 1.9.

## License

Code: MIT License (see `LICENSE`). The WDBC data are not redistributed here; they remain under CC BY 4.0.
