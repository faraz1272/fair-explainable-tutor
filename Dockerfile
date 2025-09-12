# Dockerfile (slim & fast to push)
FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# Minimal OS deps only; avoid dev toolchains unless you truly need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install CPU Torch explicitly (smallest wheel path) and then the rest
RUN python -m pip install --upgrade pip && \
    python -m pip install \
      torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt

# IMPORTANT: do NOT pre-download Detoxify or HF models here.
# (Let the app download them on first use at runtime)

# Now copy your app
COPY . .

# Streamlit port & binding for Beanstalk Docker
ENV PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8080

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]