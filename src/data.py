import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath='application_train.csv'):
    df = pd.read_csv(filepath)
    return df

def engineer_features(df):
    df = df.copy()
    # Age & Employment
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365.25
    df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'] / -365.25
    df['EMPLOYED_YEARS'] = df['EMPLOYED_YEARS'].apply(lambda x: np.nan if x < 0 else x)
    
    # Financial Ratios
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    return df

def get_features_list():
    return [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'AGE_YEARS', 'EMPLOYED_YEARS',
        'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
        'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO',
        'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS'
    ]

def split_data(df, target_col='TARGET'):
    # 3-way split: 60% train, 20% validation, 20% test
    # First split: 60% train, 40% temp (val + test)
    X = df[get_features_list()]
    y = df[target_col]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    # Second split: split the 40% temp into 20% val and 20% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
