FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --upgrade pip


RUN pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    -r requirements.txt


COPY . .

RUN mkdir -p uploads vector_store


ENV PORT=8080


EXPOSE 8080


CMD ["streamlit", "run", "src/app.py", "--server.port=8080"]
