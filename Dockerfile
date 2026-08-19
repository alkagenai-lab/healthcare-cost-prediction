FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY ml ./ml
COPY frontend ./frontend
COPY tests ./tests
COPY pytest.ini .
COPY setup.py .

# Expose both application ports
EXPOSE 8000
EXPOSE 8501