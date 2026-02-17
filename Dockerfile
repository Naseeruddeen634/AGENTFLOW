FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS production
LABEL maintainer="Naseeruddeen <nachu2003465@gmail.com>"
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/
RUN mkdir -p /app/agent_outputs
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 API_PORT=8001
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
