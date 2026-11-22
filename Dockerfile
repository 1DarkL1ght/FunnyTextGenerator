FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY src/ ./src
COPY tools/ ./tools
COPY configs/ ./configs

RUN mkdir -p outputs/logs outputs/models outputs/inference

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "tools/train.py"]
