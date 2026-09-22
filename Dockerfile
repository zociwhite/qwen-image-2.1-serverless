FROM runpod/pytorch:1.3.2-cu1290-torch291-ubuntu2404

WORKDIR /app

COPY requirements.txt .

# Debian-installed cryptography 41.0.7 has no RECORD file; pip can't upgrade it to the
# cryptography>=50 that paramiko (via runpod) requires. Force-reinstall so it becomes
# a pip-managed package, then install the rest.
RUN pip install --no-cache-dir --ignore-installed cryptography==50.0.1 && \
    pip install --no-cache-dir -r requirements.txt

# Force HF cache path & ensure permissions
ENV HF_HOME=/app/cache
ENV HF_HUB_CACHE=/app/cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/app/cache/huggingface/transformers
ENV DIFFUSERS_CACHE=/app/cache/huggingface/diffusers
ENV XDG_CACHE_HOME=/app/cache
ENV TMPDIR=/app/tmp
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN mkdir -p /app/cache /app/tmp && chmod -R 777 /app/cache /app/tmp

# Pre-download Qwen-Image-2.1 weights into image cache for 0-second cold-start
RUN python -c "from diffusers import QwenImage21Pipeline; QwenImage21Pipeline.from_pretrained('Qwen/Qwen-Image-2.1', cache_dir='/app/cache')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]