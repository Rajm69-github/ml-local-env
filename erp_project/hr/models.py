from django.db import models

class Employee(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    department = models.CharField(max_length=50)
    job_role = models.CharField(max_length=50)
    monthly_income = models.IntegerField()
    overtime = models.BooleanField(default=False)
    job_satisfaction = models.IntegerField(default=3)
    years_at_company = models.IntegerField(default=1)
    total_working_years = models.IntegerField(default=1)
    job_level = models.IntegerField(default=1)

    def __str__(self):
        return self.name