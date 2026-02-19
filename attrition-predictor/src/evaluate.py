import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_preprocess

def evaluate(data_path, model_path):
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    model = joblib.load(model_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    axes[0].set_title('Confusion Matrix')

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].set_title('ROC Curve')

    plt.tight_layout()
    plt.savefig('models/evaluation.png')
    plt.show()
    print("Saved: models/evaluation.png")

if __name__ == '__main__':
    evaluate('data/hr_attrition.csv', 'models/best_model.pkl')