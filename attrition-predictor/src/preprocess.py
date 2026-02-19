import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath, save_artifacts=True):
    df = pd.read_csv(filepath)

    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df.drop(columns=drop_cols, inplace=True)

    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if save_artifacts:
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(encoders, 'models/encoders.pkl')
        joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)