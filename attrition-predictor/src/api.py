from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Attrition Predictor API")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

class Employee(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

@app.post("/predict")
def predict(employee: Employee):
    data = pd.DataFrame([employee.dict()])

    for col, le in encoders.items():
        if col in data.columns:
            data[col] = le.transform(data[col])

    data = data[feature_columns]
    X_scaled = scaler.transform(data)

    prob = model.predict_proba(X_scaled)[0][1]
    prediction = model.predict(X_scaled)[0]

    return {
        "attrition_risk": "High" if prediction == 1 else "Low",
        "probability": round(float(prob), 4)
    }

@app.get("/health")
def health():
    return {"status": "ok"}