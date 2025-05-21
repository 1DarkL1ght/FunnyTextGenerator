FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY train.py .
COPY inference.py .
COPY data/concatenated_anekdot_dataset.csv ./data/concatenated_anekdot_dataset.csv

RUN mkdir -p outputs/logs outputs/models outputs/inference

ENV PYTHONUNBUFFERED=1
# ENV TORCH_DISTRIBUTED_DEBUG=INFO

CMD ["python", "train.py"]