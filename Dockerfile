FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir --break-system-packages onnxruntime-gpu
RUN python3.12 -m pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN python3.12 -m pip install --no-cache-dir --break-system-packages --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch torchaudio

COPY . .

CMD ["python3.12", "handler.py"]
