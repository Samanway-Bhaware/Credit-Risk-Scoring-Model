import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import make_scorer, average_precision_score

def train_xgb_cv(X_train, y_train, scale_pos_weight, n_iter=10, cv=5):
    # Base model
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Parameter space
    param_distributions = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 300, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    # Custom scorer for AUC-PR
    auc_pr_scorer = make_scorer(average_precision_score, response_method='predict_proba')
    
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=auc_pr_scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    print(f"Best params: {random_search.best_params_}")
    
    # Extract mean and std for the best score
    best_index = random_search.best_index_
    mean_score = random_search.cv_results_['mean_test_score'][best_index]
    std_score = random_search.cv_results_['std_test_score'][best_index]
    print(f"Best CV AUC-PR: {mean_score:.4f} ± {std_score:.4f}")
    
    return random_search.best_estimator_, random_search

from sklearn.isotonic import IsotonicRegression

def calibrate_model(model, X_val, y_val):
    # Calibrate using Isotonic Regression on validation set
    y_prob_val = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_prob_val, y_val)
    return calibrator
