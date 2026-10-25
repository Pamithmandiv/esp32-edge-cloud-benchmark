FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server ./server
COPY model ./model
EXPOSE 8080
CMD ["sh","-c","uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
