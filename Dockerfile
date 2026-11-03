# Serving image for the REST API.
# It reads precomputed scores (artifacts/) and the graph (data/processed/),
# both mounted at runtime, so it needs no PyTorch and no data baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY aml_gnn/ aml_gnn/
COPY api/ api/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 aml
USER aml

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
