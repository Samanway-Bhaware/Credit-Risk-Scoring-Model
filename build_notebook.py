import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    (nbf.v4.new_markdown_cell, """# Credit Risk Modelling: Decision-Support Tool

This notebook builds a credit risk model using the [Home Credit Default Risk dataset](https://www.kaggle.com/datasets/megancrenshaw/home-credit-default-risk). The goal isn't just a classifier — it's a decision-support tool that shows the trade-off between approval rate and bad-debt rate, the way a real Credit Strategy team would present it to a lending committee.

## Caveats and Limitations
- **Assumed Unit Economics:** The financial figures (profit per good loan, loss per default) are illustrative.
- **No Time-Based Validation:** The dataset lacks an application date, so a true out-of-time split isn't possible. We use stratified cross-validation and a 3-way random split as a partial substitute.
- **Calibration is Assumption-Dependent:** While we calibrate our probabilities using isotonic regression, the calibration map itself assumes the validation set perfectly represents future populations.
- **Single Population:** This model is trained on a single snapshot of data and does not account for macroeconomic shifts."""),

    (nbf.v4.new_code_cell, """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

from src.data import load_data, engineer_features, split_data
from src.features import get_preprocessor, get_feature_names
from src.train import train_xgb_cv, calibrate_model
from src.evaluate import evaluate_metrics, plot_calibration, evaluate_fairness
from src.business_metrics import simulate_thresholds, plot_business_metrics"""),

    (nbf.v4.new_markdown_cell, """## Phase 1: Data Prep & EDA

First, we load the data and check its shape, class balance, and missingness. The dataset contains various features about applicants (income, credit, age, external scores) and a `TARGET` variable indicating default (1) or repayment (0)."""),

    (nbf.v4.new_code_cell, """df_raw = load_data('application_train.csv')
print(f"Dataset shape: {df_raw.shape}")

# Class balance
target_counts = df_raw['TARGET'].value_counts(normalize=True)
print("\\nClass Balance:")
print(target_counts)

print(f"\\nDefault rate is {target_counts[1]*100:.2f}%. This is a heavily imbalanced dataset.")"""),

    (nbf.v4.new_markdown_cell, """### Missing Values Handling & Feature Engineering
Missing values are handled deliberately based on feature type (medians for numerical, "missing" category for categorical).
We create standard credit-risk features:
- **Credit-to-Income Ratio**: High ratio implies higher burden.
- **Annuity-to-Income Ratio**: A proxy for Debt-to-Income (DTI), measuring repayment capacity."""),

    (nbf.v4.new_code_cell, """df = engineer_features(df_raw)"""),

    (nbf.v4.new_markdown_cell, """## Phase 2: Modelling (3-way Split & Cross Validation)

We split the data three ways to prevent data leakage:
- **Train (60%)**: Used for model fitting and hyperparameter search via K-Fold CV.
- **Validation (20%)**: Used for calibration, threshold selection, and business-cost analysis.
- **Test (20%)**: Used for final unbiased evaluation, touched exactly once, reported as-is.

*Note: Since a true out-of-time split isn't available, we use stratified k-fold cross-validation on the training set to evaluate score stability.*"""),

    (nbf.v4.new_code_cell, """X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

preprocessor, num_cols, cat_cols = get_preprocessor(X_train)
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

feature_names = get_feature_names(preprocessor, num_cols, cat_cols)"""),

    (nbf.v4.new_markdown_cell, """### Hyperparameter Search
We use `RandomizedSearchCV` over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, and `colsample_bytree`, scored on cross-validated AUC-PR (not AUC-ROC, as PR is the more honest metric under this imbalance)."""),

    (nbf.v4.new_code_cell, """# Calculate class weight ratio: negative cases / positive cases
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

best_xgb, search_results = train_xgb_cv(X_train_proc, y_train, scale_pos_weight=scale_pos_weight, n_iter=10, cv=5)"""),

    (nbf.v4.new_markdown_cell, """## Phase 3: Calibration

We plot a calibration curve (reliability diagram) for the raw XGBoost output on the validation set, alongside its Brier score.
Then, we apply **Isotonic Regression** (using `CalibratedClassifierCV`) and re-plot.

*Note: `scale_pos_weight` reweights the loss function to fix ranking under imbalance, while calibration makes the output probability-scale-correct so that 10% actually means 10%.*"""),

    (nbf.v4.new_code_cell, """val_preds_raw = best_xgb.predict_proba(X_val_proc)[:, 1]
evaluate_metrics(y_val, val_preds_raw, dataset_name="Validation (Raw)")

calibrator = calibrate_model(best_xgb, X_val_proc, y_val)
val_preds_cal = calibrator.predict(val_preds_raw)
evaluate_metrics(y_val, val_preds_cal, dataset_name="Validation (Calibrated)")

_ = plot_calibration(y_val, val_preds_raw, val_preds_cal, title_suffix="(Validation)")"""),

    (nbf.v4.new_markdown_cell, """## Phase 4: The Business Framing (Validation Set)

Using the **calibrated probabilities**, we sweep the decision threshold from 0 to 1 and calculate Approval Rate vs Bad Rate.

### Expected Value Framing
We assume:
- Profit per good loan: **$1,000**
- Loss per bad loan: **-$5,000**"""),

    (nbf.v4.new_code_cell, """df_results = simulate_thresholds(y_val, val_preds_cal)
fig_biz, max_ev_row = plot_business_metrics(df_results, title_suffix="(Validation, Calibrated)")

optimal_threshold = max_ev_row['Threshold']
print(f"Optimal Threshold from Validation: {optimal_threshold:.4f} (Expected Value: ${max_ev_row['Expected Value']:,.2f})")"""),

    (nbf.v4.new_markdown_cell, """## Phase 5: Fairness Audit

We use `CODE_GENDER` as group labels (not model inputs) to compute approval rate, false-positive rate, false-negative rate, and default rate among approved applicants, at our chosen threshold.

We also compute the **Disparate Impact Ratio** (approval rate of lower group / higher group) and note whether it clears the 80% "four-fifths rule" reference point."""),

    (nbf.v4.new_code_cell, """df_val_fairness = df.loc[X_val.index]
fairness_results = evaluate_fairness(df_val_fairness, y_val.values, val_preds_cal, optimal_threshold)
display(fairness_results)

approval_rates = fairness_results['Approval Rate']
if len(approval_rates) >= 2:
    dir_ratio = approval_rates.min() / approval_rates.max() if approval_rates.max() > 0 else 0
    print(f"\\nDisparate Impact Ratio: {dir_ratio:.4f}")"""),

    (nbf.v4.new_markdown_cell, """## Phase 6: Final Test Set Evaluation

We run the final chosen threshold once on the test set and report that as the headline number."""),

    (nbf.v4.new_code_cell, """print("\\nFinal Unbiased Evaluation on Test Set...")
test_preds_raw = best_xgb.predict_proba(X_test_proc)[:, 1]
test_preds_cal = calibrator.predict(test_preds_raw)
evaluate_metrics(y_test, test_preds_cal, dataset_name="Test (Calibrated)")

n_total_test = len(y_test)
approved_test = test_preds_cal < optimal_threshold
n_approve_test = approved_test.sum()
approval_rate_test = n_approve_test / n_total_test
bad_rate_test = y_test[approved_test].mean() if n_approve_test > 0 else 0

goods = n_approve_test - y_test[approved_test].sum()
bads = y_test[approved_test].sum()
ev_test = (goods * 1000) + (bads * -5000)

print(f"\\nTest Set Business Outcome (using Validation Optimal Threshold {optimal_threshold:.4f}):")
print(f"Approval Rate: {approval_rate_test*100:.1f}%")
print(f"Bad Rate:      {bad_rate_test*100:.1f}%")
print(f"Expected Val:  ${ev_test:,.2f}")""")
]

for func, content in cells:
    nb.cells.append(func(content))

with open("Credit_Risk_Analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
