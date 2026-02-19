import numpy as np
import joblib
import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_preprocess

def train(data_path, model_path):
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    mlflow.set_experiment("attrition-predictor")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    best_model, best_auc = None, 0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            report = classification_report(y_test, y_pred, output_dict=True)

            # Log params & metrics
            mlflow.log_param("model", name)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("precision_1", report['1']['precision'])
            mlflow.log_metric("recall_1", report['1']['recall'])
            mlflow.log_metric("f1_1", report['1']['f1-score'])

            # Log model
            mlflow.sklearn.log_model(model, name)

            print(f"\n--- {name} --- AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_model = model

    joblib.dump(best_model, model_path)
    print(f"\nBest model saved: {type(best_model).__name__} (AUC: {best_auc:.4f})")

if __name__ == '__main__':
    train('data/hr_attrition.csv', 'models/best_model.pkl')