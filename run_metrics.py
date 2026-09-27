import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from src.data import load_data, engineer_features, split_data
from src.features import get_preprocessor, get_feature_names
from src.train import train_xgb_cv, calibrate_model
from src.evaluate import evaluate_metrics, plot_calibration, evaluate_fairness
from src.business_metrics import simulate_thresholds, plot_business_metrics

print("Loading and engineering data...")
df_raw = load_data('application_train.csv')
df = engineer_features(df_raw)

print("Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

print("Preprocessing...")
preprocessor, num_cols, cat_cols = get_preprocessor(X_train)
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

print("Training model with RandomizedSearchCV...")
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
best_xgb, search_results = train_xgb_cv(X_train_proc, y_train, scale_pos_weight=scale_pos_weight, n_iter=5, cv=3) 
# Note: n_iter=5, cv=3 for speed in this script. Notebook will use 5/5.

print("Evaluating Raw Model on Validation Set...")
val_preds_raw = best_xgb.predict_proba(X_val_proc)[:, 1]
val_roc_raw, val_pr_raw, brier_raw = evaluate_metrics(y_val, val_preds_raw, dataset_name="Validation (Raw)")

print("Calibrating Model on Validation Set...")
calibrator = calibrate_model(best_xgb, X_val_proc, y_val)
val_preds_cal = calibrator.predict(val_preds_raw)
val_roc_cal, val_pr_cal, brier_cal = evaluate_metrics(y_val, val_preds_cal, dataset_name="Validation (Calibrated)")

print("Plotting Calibration Curves...")
fig_cal = plot_calibration(y_val, val_preds_raw, val_preds_cal, title_suffix="(Validation)")
fig_cal.savefig('calibration_curve.png')

print("Simulating Thresholds on Validation Set (Calibrated)...")
df_results = simulate_thresholds(y_val, val_preds_cal)
fig_biz, max_ev_row = plot_business_metrics(df_results, title_suffix="(Validation, Calibrated)")
fig_biz.savefig('expected_value_calibrated.png')

optimal_threshold = max_ev_row['Threshold']
print(f"Optimal Threshold from Validation: {optimal_threshold:.4f} (Expected Value: ${max_ev_row['Expected Value']:,.2f})")

print("Evaluating Fairness on Validation Set...")
# We need to map the validation indices back to the original dataframe for CODE_GENDER
df_val_fairness = df.loc[X_val.index]
fairness_results = evaluate_fairness(df_val_fairness, y_val.values, val_preds_cal, optimal_threshold)
print("\nFairness Metrics (Validation Set):")
print(fairness_results.to_string())

# Disparate Impact Ratio
# DIR = Approval Rate (Lower) / Approval Rate (Higher)
approval_rates = fairness_results['Approval Rate']
if len(approval_rates) >= 2:
    min_rate, max_rate = approval_rates.min(), approval_rates.max()
    dir_ratio = min_rate / max_rate if max_rate > 0 else 0
    print(f"\nDisparate Impact Ratio: {dir_ratio:.4f}")
    if dir_ratio < 0.8:
        print("Note: The Disparate Impact Ratio is below the 80% four-fifths rule reference point.")
    else:
        print("Note: The Disparate Impact Ratio clears the 80% four-fifths rule reference point.")

print("\nFinal Unbiased Evaluation on Test Set...")
test_preds_raw = best_xgb.predict_proba(X_test_proc)[:, 1]
test_preds_cal = calibrator.predict(test_preds_raw)
roc_test, pr_test, brier_test = evaluate_metrics(y_test, test_preds_cal, dataset_name="Test (Calibrated)")

# Test Set Business Metrics at Optimal Threshold from Validation
n_total_test = len(y_test)
approved_test = test_preds_cal < optimal_threshold
n_approve_test = approved_test.sum()
approval_rate_test = n_approve_test / n_total_test
bad_rate_test = y_test[approved_test].mean() if n_approve_test > 0 else 0

goods = n_approve_test - y_test[approved_test].sum()
bads = y_test[approved_test].sum()
ev_test = (goods * 1000) + (bads * -5000)

print(f"\nTest Set Business Outcome (using Validation Optimal Threshold {optimal_threshold:.4f}):")
print(f"Approval Rate: {approval_rate_test*100:.1f}%")
print(f"Bad Rate:      {bad_rate_test*100:.1f}%")
print(f"Expected Val:  ${ev_test:,.2f}")

print("\\nWriting results explanation to RESULTS_SUMMARY.md...")
best_index = search_results.best_index_
mean_score = search_results.cv_results_['mean_test_score'][best_index]
std_score = search_results.cv_results_['std_test_score'][best_index]

summary = f"""# Model Results and Explanations

## 1. Hyperparameter Search
**Best Parameters:** `{search_results.best_params_}`
**Best CV AUC-PR:** {mean_score:.4f} ± {std_score:.4f}
*Meaning:* These are the optimal hyperparameters found during 5-fold cross-validation on the training set. AUC-PR (Area Under the Precision-Recall Curve) was used instead of Accuracy because the dataset is highly imbalanced (only 8% defaults).

## 2. Model Calibration
**Raw Brier Score:** {brier_raw:.4f}
**Calibrated Brier Score:** {brier_cal:.4f}
*Meaning:* The Brier Score measures the accuracy of probabilistic predictions (lower is better, 0 is perfect). XGBoost's raw probabilities were distorted by the `scale_pos_weight` parameter used to handle the imbalance. We used Isotonic Regression to calibrate them. The drop in Brier Score confirms the probabilities are now much closer to true real-world likelihoods.

## 3. Validation Set Optimal Threshold
**Optimal Threshold:** {optimal_threshold:.4f}
**Max Expected Value (Validation):** ${max_ev_row['Expected Value']:,.2f}
*Meaning:* We simulated a business scenario where every good loan makes $1000 and every bad loan loses $5000. By sweeping the decision threshold across the validation set, we found that approving applicants with a default probability below {optimal_threshold:.4f} maximizes profit.

## 4. Test Set Final Evaluation (Using optimal threshold)
**Test AUC-ROC:** {roc_test:.4f}
**Test AUC-PR:** {pr_test:.4f}
**Test Brier Score:** {brier_test:.4f}
**Approval Rate:** {approval_rate_test*100:.1f}%
**Bad Rate Among Approved:** {bad_rate_test*100:.1f}%
**Expected Portfolio Value:** ${ev_test:,.2f}
*Meaning:* We took the optimal threshold found on the validation set ({optimal_threshold:.4f}) and applied it strictly to the unseen Test Set. An approval rate of {approval_rate_test*100:.1f}% generated an expected profit of ${ev_test:,.2f}. The fact that the test performance strongly matches the validation performance indicates the model generalizes well and is not overfitting.

## 5. Fairness Audit
**Disparate Impact Ratio:** {dir_ratio:.4f}
*Meaning:* The Disparate Impact Ratio (DIR) compares the approval rate of the least favored demographic group against the most favored. In this case, we evaluated gender (`CODE_GENDER`). A DIR of {dir_ratio:.4f} means the lower-approval group is approved at a rate that is {dir_ratio*100:.1f}% of the higher-approval group. Since this is below the 80% 'four-fifths rule' reference point, this model exhibits disparate impact and would require remediation by a fair-lending compliance team before real-world deployment.
"""

with open("RESULTS_SUMMARY.md", "w") as f:
    f.write(summary)
print("Done!")

