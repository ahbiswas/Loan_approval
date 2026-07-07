# README.md

## Project overview

This repo is a single Jupyter notebook (`LoanDefaultProbability.ipynb`) that builds a binary classifier predicting whether a LendingClub loan will charge off (default) vs. be fully paid. There is no application code, package, or test suite — all logic lives in the notebook's cells, executed top to bottom.

- `LoanDefaultProbability.ipynb` — the entire pipeline: data loading, feature selection, feature engineering, categorical encoding, model comparison, tuning, and final model export.
- `LoansData_sample.csv` — sample of the LendingClub loans dataset (subset of the ~1GB full dataset referenced on Kaggle at `mlfinancebook/lending-club-loans-data`). The notebook is written to also accept the full gzipped dataset (`LoansData.csv.gz`) via a commented-out load line.
- `requirements.txt` — flat dependency list (numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow, keras, scikeras, joblib).
- `.venv` — local virtualenv (Python 3.11).

## Setup and running

```bash
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook LoanDefaultProbability.ipynb
```

There are no lint, test, or build commands — this is an exploratory notebook, not a package. To validate changes, re-run the notebook end-to-end (`Kernel > Restart & Run All`) and check that later cells (model comparison, final GBM fit, artifact export) still execute without error.

To inspect or edit a notebook's cells programmatically without opening Jupyter, load it as JSON:

```python
import json
nb = json.load(open('LoanDefaultProbability.ipynb'))
nb['cells'][i]['source']  # list of source lines for cell i
```

## Pipeline architecture

The notebook executes as one linear sequence; later cells depend on `dataset`, `X_train`/`Y_train`, etc. mutated in earlier cells. Key stages, in order:

1. **Load** — read `LoansData_sample.csv` into `dataset`.
2. **Define target** — keep only loans with `loan_status` in `{Fully Paid, Charged Off}`; derive binary target `charged_off` and drop `loan_status`.
3. **Feature selection** (three passes, each shrinking `dataset.columns`):
   - Drop columns with >30% missing values.
   - Drop columns deemed unintuitive/unavailable to investors at loan-issue time (keeps an explicit `keep_list`).
   - Drop numeric columns with `|corr(feature, charged_off)| < 0.03`.
4. **Feature engineering** — per-feature cleanup done by hand (e.g. `term` string → int months, `emp_length` parsed then dropped as uninformative, `earliest_cr_line` → year, `annual_inc` → `log_annual_inc`, `fico_range_low/high` averaged into `fico_score`).
5. **Categorical encoding** — categorical columns are detected via `select_dtypes(include=['object','string','category'])` (not just `object`, since pandas' newer string dtype can be missed — see the comment above that cell) and label-encoded in place, fitting one `sklearn.LabelEncoder` per column and keeping them in an `encoders` dict (a single shared encoder instance would only retain the fit for the last column processed).
6. **Dimensionality reduction** — standardize the encoded features with `StandardScaler`, then fit `PCA(n_components=0.95, random_state=7)` and replace the feature columns in `dataset` with the resulting principal components (`pc_1`, `pc_2`, …), keeping `charged_off`.
7. **Class balancing** — undersample both classes to 5,500 rows each (dataset is naturally ~79/21 imbalanced) and shuffle.
8. **Train/validation split** — `train_test_split` with `test_size=0.2, random_state=7`.
9. **Model comparison** — 10-fold CV (`scoring='roc_auc'`) across LR, LDA, KNN, CART, GaussianNB, MLPClassifier, AdaBoost, GradientBoosting, RandomForest, ExtraTrees. On the PCA-reduced features, Logistic Regression currently wins (ROC AUC ≈ 0.91) — pick whichever model tops this comparison, since PCA's linear/decorrelated components can shift which model is best.
10. **Tuning** — `GridSearchCV` over the winning model from step 9 (currently `LogisticRegression`, tuning `C` and `penalty` via the `liblinear` solver).
11. **Finalize** — refit the tuned model (`LogisticRegression(**grid_result.best_params_, ...)`) on the full training split, evaluate on the validation split (accuracy, confusion matrix, classification report), and inspect feature importance — `feature_importances_` for tree-based models, or `coef_` for linear models like the current one.
12. **Export artifacts** — `joblib.dump` the fitted model to `best_model.joblib`, the per-column `LabelEncoder`s to `label_encoders.joblib`, and the `StandardScaler`/`PCA` objects to `pca_scaler.joblib`/`pca_transform.joblib`, so new raw data can be preprocessed identically (encode → scale → PCA transform) at inference time.

When modifying the pipeline, keep the encoders/scaler/PCA and the final model in sync: any change to `categorical_cols`, `keep_list`, or the PCA variance threshold upstream must be reflected in what's persisted in the corresponding `.joblib` file, since inference code outside this notebook would rely on all of them matching. If you swap in a different winning model at step 9–10, update step 11's feature-importance approach accordingly (`feature_importances_` vs `coef_`).
