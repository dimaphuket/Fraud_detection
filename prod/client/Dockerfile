FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client.py .

# Директория для данных (монтируется через volume)
RUN mkdir -p /app/data

# Интерактивный режим
ENV PYTHONUNBUFFERED=1

CMD ["python", "client.py"]
