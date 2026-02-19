FROM python:3.10-slim

WORKDIR /app

RUN pip install django requests

COPY . .

EXPOSE 8080

CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]
