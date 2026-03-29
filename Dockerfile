# ─────────────────────────────────────────────────────────────
# Clinical Triage OpenEnv — Dockerfile
# ─────────────────────────────────────────────────────────────
# Build:  docker build -t clinical-triage-env .
# Run:    docker run -p 7860:7860 clinical-triage-env
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="clinical-triage-openenv"
LABEL description="Clinical Triage OpenEnv — HF Spaces compatible"
LABEL version="1.0.0"

# Create a non-root user (security best practice)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Give ownership to app user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port expected by Hugging Face Spaces
EXPOSE 7860

# Health check so HF Spaces knows the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]