import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

def evaluate_metrics(y_true, y_prob, dataset_name=""):
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    print(f"--- {dataset_name} Evaluation ---")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"Brier:   {brier:.4f}")
    return auc_roc, auc_pr, brier

def plot_calibration(y_true, y_prob_raw, y_prob_calibrated, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Raw model
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=10)
    ax.plot(prob_pred_raw, prob_true_raw, marker='o', linewidth=1, label='XGBoost (Raw)')
    
    # Calibrated model
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
    ax.plot(prob_pred_cal, prob_true_cal, marker='s', linewidth=1, label='XGBoost (Isotonic Calibrated)')
    
    # Perfectly calibrated
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
    
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration Curve {title_suffix}')
    ax.legend()
    plt.grid(True)
    plt.show()
    return fig

def evaluate_fairness(df_val, y_true, y_prob, threshold, group_col='CODE_GENDER'):
    results = []
    groups = df_val[group_col].unique()
    
    for group in groups:
        mask = df_val[group_col] == group
        y_true_g = y_true[mask]
        y_prob_g = y_prob[mask]
        
        n_total = len(y_true_g)
        if n_total == 0:
            continue
            
        approved = y_prob_g < threshold
        n_approved = approved.sum()
        approval_rate = n_approved / n_total
        
        # True negatives are actual defaults (1) that were rejected (prob >= threshold)
        # Wait, the threshold is on probability of default.
        # Below threshold = approved (predicted 0). Above = rejected (predicted 1).
        # False Positive (FPR): Actual Good (0) but Rejected (pred 1)
        # False Negative (FNR): Actual Bad (1) but Approved (pred 0)
        
        # Let's clarify: Positive class = Default (1).
        # FPR: True = 0, Pred = 1 (rejected). So FPR is rejection rate among good loans.
        # FNR: True = 1, Pred = 0 (approved). So FNR is approval rate among bad loans.
        actual_goods = (y_true_g == 0)
        actual_bads = (y_true_g == 1)
        
        fpr = (~approved[actual_goods]).mean() if actual_goods.sum() > 0 else 0
        fnr = (approved[actual_bads]).mean() if actual_bads.sum() > 0 else 0
        
        # Default rate among approved
        bad_rate = y_true_g[approved].mean() if n_approved > 0 else 0
        
        results.append({
            'Group': group,
            'N': n_total,
            'Approval Rate': approval_rate,
            'FPR (Good but Rejected)': fpr,
            'FNR (Bad but Approved)': fnr,
            'Default Rate (Among Approved)': bad_rate
        })
        
    df_fairness = pd.DataFrame(results).set_index('Group')
    return df_fairness
