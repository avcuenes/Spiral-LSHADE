FROM python:3.11-slim

LABEL maintainer="Spiral-LSHADE contributors"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System packages required to compile scientific Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Default to an interactive shell so users can invoke any of the Make targets.
CMD ["/bin/bash"]
