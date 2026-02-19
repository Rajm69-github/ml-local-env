from django.shortcuts import render, get_object_or_404
from .models import Employee
from .services.attrition_service import predict_attrition

def employee_list(request):
    employees = Employee.objects.all()
    return render(request, 'hr/employee_list.html', {'employees': employees})

def employee_detail(request, pk):
    employee = get_object_or_404(Employee, pk=pk)

    payload = {
        "Age": employee.age,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 500,
        "Department": employee.department,
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 50,
        "JobInvolvement": 3,
        "JobLevel": employee.job_level,
        "JobRole": employee.job_role,
        "JobSatisfaction": employee.job_satisfaction,
        "MaritalStatus": "Single",
        "MonthlyIncome": employee.monthly_income,
        "MonthlyRate": 10000,
        "NumCompaniesWorked": 2,
        "OverTime": "Yes" if employee.overtime else "No",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 0,
        "TotalWorkingYears": employee.total_working_years,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": employee.years_at_company,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 1
    }

    result = predict_attrition(payload)
    return render(request, 'hr/employee_detail.html', {
        'employee': employee,
        'prediction': result
    })