# syntax=docker/dockerfile:1
# ---- Etape 1 : installation des dependances + entrainement du modele ----
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.api.txt
COPY . .
ENV PYTHONPATH=/install/lib/python3.12/site-packages
RUN python train.py

# ---- Etape finale : image d'execution ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
COPY --from=builder /build/artifacts/ticket_model.joblib ./artifacts/ticket_model.joblib
COPY classifier.py ./classifier.py
COPY artifacts/ticket_model.joblib ./artifacts/ticket_model.joblib
COPY api.py .
ENV MODEL_PATH="artifacts/ticket_model.joblib"
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
