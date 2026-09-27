# Model Results and Explanations

## 1. Hyperparameter Search
**Best Parameters:** `{'subsample': 0.6, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.6}`
**Best CV AUC-PR:** 0.2230 ± 0.0067
*Meaning:* These are the optimal hyperparameters found during 5-fold cross-validation on the training set. AUC-PR (Area Under the Precision-Recall Curve) was used instead of Accuracy because the dataset is highly imbalanced (only 8% defaults).

## 2. Model Calibration
**Raw Brier Score:** 0.2000
**Calibrated Brier Score:** 0.0680
*Meaning:* The Brier Score measures the accuracy of probabilistic predictions (lower is better, 0 is perfect). XGBoost's raw probabilities were distorted by the `scale_pos_weight` parameter used to handle the imbalance. We used Isotonic Regression to calibrate them. The drop in Brier Score confirms the probabilities are now much closer to true real-world likelihoods.

## 3. Validation Set Optimal Threshold
**Optimal Threshold:** 0.1744
**Max Expected Value (Validation):** $35,290,000.00
*Meaning:* We simulated a business scenario where every good loan makes $1000 and every bad loan loses $5000. By sweeping the decision threshold across the validation set, we found that approving applicants with a default probability below 0.1744 maximizes profit.

## 4. Test Set Final Evaluation (Using optimal threshold)
**Test AUC-ROC:** 0.7514
**Test AUC-PR:** 0.2324
**Test Brier Score:** 0.0681
**Approval Rate:** 89.1%
**Bad Rate Among Approved:** 5.9%
**Expected Portfolio Value:** $35,357,000.00
*Meaning:* We took the optimal threshold found on the validation set (0.1744) and applied it strictly to the unseen Test Set. An approval rate of 89.1% generated an expected profit of $35,357,000.00. The fact that the test performance strongly matches the validation performance indicates the model generalizes well and is not overfitting.

## 5. Fairness Audit
**Disparate Impact Ratio:** 0.5437
*Meaning:* The Disparate Impact Ratio (DIR) compares the approval rate of the least favored demographic group against the most favored. In this case, we evaluated gender (`CODE_GENDER`). A DIR of 0.5437 means the lower-approval group is approved at a rate that is 54.4% of the higher-approval group. Since this is below the 80% 'four-fifths rule' reference point, this model exhibits disparate impact and would require remediation by a fair-lending compliance team before real-world deployment.
