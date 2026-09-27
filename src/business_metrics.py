import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_thresholds(y_true, y_prob, profit_good=1000, loss_bad=-5000):
    df_test = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values(by='y_prob')
    n_total = len(df_test)
    
    percentiles = np.linspace(0.01, 1.0, 100)
    results = []
    
    for p in percentiles:
        n_approve = int(p * n_total)
        if n_approve == 0:
            continue
            
        approved_subset = df_test.iloc[:n_approve]
        
        approval_rate = n_approve / n_total
        bad_rate = approved_subset['y_true'].sum() / n_approve
        threshold = approved_subset['y_prob'].iloc[-1]
        
        goods = n_approve - approved_subset['y_true'].sum()
        bads = approved_subset['y_true'].sum()
        
        ev = (goods * profit_good) + (bads * loss_bad)
        
        results.append({
            'Percentile': p,
            'Threshold': threshold,
            'Approval Rate': approval_rate,
            'Bad Rate': bad_rate,
            'Expected Value': ev
        })
        
    return pd.DataFrame(results)

def plot_business_metrics(df_results, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Approval vs Bad Rate
    axes[0].plot(df_results['Approval Rate'], df_results['Bad Rate'], label='Bad Rate Curve', color='blue', linewidth=2)
    axes[0].set_xlabel('Approval Rate')
    axes[0].set_ylabel('Bad Rate Among Approved')
    axes[0].set_title(f'Business Trade-off {title_suffix}')
    axes[0].grid(True)
    
    # Expected Value
    axes[1].plot(df_results['Approval Rate'], df_results['Expected Value'], color='green', linewidth=2)
    
    # Max EV line
    max_ev_idx = df_results['Expected Value'].idxmax()
    max_ev_row = df_results.loc[max_ev_idx]
    axes[1].axvline(x=max_ev_row['Approval Rate'], color='red', linestyle='--', 
                    label=f'Max EV at {max_ev_row["Approval Rate"]*100:.0f}% Approval')
    
    axes[1].set_xlabel('Approval Rate')
    axes[1].set_ylabel('Expected Portfolio Value ($)')
    axes[1].set_title(f'Expected Portfolio Value {title_suffix}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    return fig, max_ev_row
