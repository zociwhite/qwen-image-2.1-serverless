# Qwen-Image-2.1 Serverless (RunPod)

RunPod serverless worker for [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
via `diffusers.QwenImage21Pipeline`.

- **Text-to-image**: omit `images` → `prompt` + optional `width`/`height`.
- **Image editing**: pass `images` (1–10, base64 / data-URI / URL). For 1 image also `image`.
- **RGBA transparency** supported by the model.

## Input (`{"input": {...}}`)

| Key | Wajib | Default | Catatan |
|---|---|---|---|
| `prompt` | ✅ | – | deskripsi |
| `images` | – | [] | maks 10 ref gambar (edit mode). base64 RAW, data URI, atau URL |
| `image` | – | – | atajo: ref gambar pertama (edit mode) |
| `negative_prompt` | – | – | |
| `width` / `height` | – | 1024 / 1024 | T2I; hijauat gan jaribu (2048 max) |
| `steps` | – | 40 | |
| `guidance_scale` | – | – | pipeline default |
| `seed` | – | random | |
| `output_format` | – | png | png \| jpeg \| webp |

## Output

```json
{ "image": "<base64 RAW tanpa prefix>", "format": "png" }
```

UI normalization: `data:image/<format>;base64,` + `image`.

## Dockerfile

- Base: `runpod/pytorch:1.3.2-cu1290-torch291-ubuntu2404` (torch 2.9.1 preinstalled → fast build).
- Env: `MODEL_OFFLOAD=1` to enable model CPU offload (24GB VRAM, large output).
- Deploy via RunPod **Import Git Repository** from this repo.