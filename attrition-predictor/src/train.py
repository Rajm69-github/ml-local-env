import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_preprocess

def train(data_path, model_path):
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    best_model, best_auc = None, 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Save best model
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved: {type(best_model).__name__} (AUC: {best_auc:.4f})")

if __name__ == '__main__':
    train('data/hr_attrition.csv', 'models/best_model.pkl')