# Employee Attrition Predictor

End-to-end ML pipeline to predict employee attrition using IBM HR dataset.

## Stack
- Python 3.10, scikit-learn, pandas, matplotlib, seaborn, joblib

## Project Structure
```
attrition-predictor/
├── data/               # Raw dataset
├── notebooks/          # EDA notebook
├── src/
│   ├── preprocess.py   # Data cleaning & feature engineering
│   ├── train.py        # Model training & selection
│   └── evaluate.py     # Confusion matrix & ROC curve
├── models/             # Saved model + evaluation plots
└── requirements.txt
```

## Setup
```bash
conda create -n ml-local python=3.10 -y
conda activate ml-local
pip install -r requirements.txt
```

## Run
```bash
# Train
python src/train.py

# Evaluate
python src/evaluate.py
```

## Results
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.8059 |
| Random Forest | 0.7648 |

**Best model:** Logistic Regression — saved to `models/best_model.pkl`