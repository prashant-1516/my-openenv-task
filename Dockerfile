Copy

FROM python:3.11-slim-bookworm
 
WORKDIR /app
 
# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
 
# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
# Copy all project files
COPY . .
 
EXPOSE 7860
 
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1
 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
