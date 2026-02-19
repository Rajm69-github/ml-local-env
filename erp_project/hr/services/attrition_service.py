import requests

ATTRITION_API_URL = "http://127.0.0.1:8000/predict"

def predict_attrition(employee_data: dict) -> dict:
    try:
        response = requests.post(
            ATTRITION_API_URL,
            json=employee_data,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "ML service unavailable"}
    except requests.exceptions.Timeout:
        return {"error": "ML service timed out"}
    except Exception as e:
        return {"error": str(e)}