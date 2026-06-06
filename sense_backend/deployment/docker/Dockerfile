# SENSE Backend — container image
# Build from project root:
#   docker build -f deployment/docker/Dockerfile -t sense-backend .
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; AI/audio libs are the pipeline's concern.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install backend requirements first (better layer caching)
COPY backend/requirements-backend.txt /app/backend/requirements-backend.txt
RUN pip install --no-cache-dir -r backend/requirements-backend.txt

# Copy the project
COPY . /app

ENV SENSE_HOST=0.0.0.0 \
    SENSE_PORT=8000 \
    SENSE_PIPELINE_MODE=hybrid

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
