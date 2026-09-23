FROM runpod/pytorch:1.3.2-cu1290-torch291-ubuntu2404

WORKDIR /app

COPY requirements.txt .

# Debian-installed cryptography 41.0.7 has no RECORD file; pip can't upgrade it to cryptography>=50
RUN pip install --no-cache-dir --ignore-installed cryptography==50.0.1 && \
    pip install --no-cache-dir -r requirements.txt

# Environment variables for HF cache and PyTorch allocator
ENV HF_HOME=/app/cache
ENV HF_HUB_CACHE=/app/cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/app/cache/huggingface/transformers
ENV DIFFUSERS_CACHE=/app/cache/huggingface/diffusers
ENV XDG_CACHE_HOME=/app/cache
ENV TMPDIR=/app/tmp
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN mkdir -p /app/cache /app/tmp && chmod -R 777 /app/cache /app/tmp

COPY handler.py .

CMD ["python", "-u", "handler.py"]
