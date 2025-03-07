# Use a lightweight Python base image
FROM python:3.10-slim

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install dependencies
COPY ./requirements.txt /app/core/requirements.txt

RUN pip install -r /app/core/requirements.txt

# Copy all necessary application files
COPY . /app
ENV PYTHONPATH="/app"

# Expose the public-facing port for Fly.io
EXPOSE 8000

# Start only core and text_analysis services
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]
