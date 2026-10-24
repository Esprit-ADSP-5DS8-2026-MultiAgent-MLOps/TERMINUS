# syntax=docker/dockerfile:1
# ---- Etape finale : image d'execution ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home --uid 1000 appuser
COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt
COPY main.py ./main.py
COPY iris_model.pkl ./iris_model.pkl
COPY api.py .
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
