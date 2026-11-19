
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY saved_models/ ./saved_models/

EXPOSE 7860 8000

CMD ["python", "app.py"]
