FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_APP=api/app.py

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create db directory and set permissions
RUN mkdir -p /app/db && chmod 777 /app/db
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN chmod 777 /app/db
# Expose port
EXPOSE 5009

# Start Gunicorn with eventlet worker
CMD ["gunicorn", "--worker-class", "eventlet", "--workers", "1", "--bind", "0.0.0.0:5009", "--timeout", "120", "--keep-alive", "5", "--log-level", "info", "api.app:app"]