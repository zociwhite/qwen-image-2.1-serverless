"""RunPod serverless worker for Qwen/Qwen-Image-2.1.

Text-to-image (no images input) or image editing (1-10 reference images).
Returns RAW base64 (no data: prefix) in {"image": "...", "format": "png"}.
"""
import base64
import io
import os
import re
import shutil

import requests
import runpod
import torch
from PIL import Image

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1")
_MAX_IMAGES = 10

# Force HF cache + tempfile into /app (150GB containerDisk-backed, NOT /workspace or /tmp).
os.environ["HF_HOME"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache/huggingface/transformers"
os.environ["DIFFUSERS_CACHE"] = "/app/cache/huggingface/diffusers"
os.environ["XDG_CACHE_HOME"] = "/app/cache"
os.environ["TMPDIR"] = "/app/tmp"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.makedirs("/app/cache", exist_ok=True)
os.makedirs("/app/tmp", exist_ok=True)

_pipe = None


def _load_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir="/app/cache",
        )
        if os.environ.get("MODEL_OFFLOAD") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
        _pipe = pipe
    return _pipe


def _decode_image(src):
    if isinstance(src, dict):
        return _decode_image(src.get("image") or src.get("url") or src.get("base64"))
    if not isinstance(src, str):
        raise ValueError("image must be base64, data URI, or URL")
    if re.match(r"^https?://", src):
        resp = requests.get(src, timeout=120)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if src.startswith("data:"):
        src = src.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(src))).convert("RGB")


def _to_b64(img, fmt):
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    return base64.b64encode(buf.getvalue()).decode()


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def handler(job):
    job_input = job.get("input") or {}
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("'prompt' is required")

    pipe = _load_pipe()

    images_raw = list(job_input.get("images") or [])
    if job_input.get("image"):
        images_raw.insert(0, job_input["image"])
    images = [_decode_image(i) for i in images_raw[:_MAX_IMAGES]]

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": _as_int(job_input.get("steps"), 20),
    }
    if job_input.get("negative_prompt"):
        kwargs["negative_prompt"] = str(job_input["negative_prompt"])
    if job_input.get("guidance_scale"):
        kwargs["true_cfg_scale"] = float(job_input["guidance_scale"])
    elif job_input.get("true_cfg_scale"):
        kwargs["true_cfg_scale"] = float(job_input["true_cfg_scale"])
    seed = job_input.get("seed")
    if seed is not None:
        kwargs["generator"] = torch.Generator("cuda").manual_seed(_as_int(seed, 0))

    if images:
        kwargs["image"] = images[0] if len(images) == 1 else images
        if job_input.get("width") and job_input.get("height"):
            kwargs["width"] = _as_int(job_input["width"], images[0].width)
            kwargs["height"] = _as_int(job_input["height"], images[0].height)
    else:
        kwargs["width"] = _as_int(job_input.get("width"), 1024)
        kwargs["height"] = _as_int(job_input.get("height"), 1024)

    out = pipe(**kwargs).images[0]

    fmt = (job_input.get("output_format") or "png").lower()
    if fmt not in ("png", "jpeg", "webp"):
        fmt = "png"
    return {"image": _to_b64(out, fmt), "format": fmt}


runpod.serverless.start({"handler": handler})