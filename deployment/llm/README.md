# LLM inference backends

Runs a generative model as one more member of the ensemble, on the same
`POST /inference` contract as the CNN services.

Two pieces:

| Piece | Where | What it is |
| --- | --- | --- |
| Adapter | `src/llm_service/` | FastAPI service on port 5012. Accepts an image, calls the engine, returns `{"response": ...}`. |
| Engine | this folder | The model server itself — Ollama or vLLM — as a compose overlay. |

The adapter speaks both engines, selected by `LLM_BACKEND`:

- `ollama` — Ollama's native API, images sent as raw bytes.
- `openai` — any OpenAI-compatible `/v1/chat/completions` server (vLLM, or Ollama's own `/v1` shim), images sent as a base64 `data:` URL.

So swapping engines is an environment change, not a code change.

## Ollama (default, works on CPU)

```bash
docker compose -f src/docker-compose.yml \
               -f deployment/llm/docker-compose.ollama.yaml up
```

This starts Ollama in a container, pulls the model into a named volume, and only
then starts the adapter — so the first request never races an unpulled model.
Weights survive `docker compose down`.

Change the model without rebuilding anything:

```bash
LLM_MODEL=qwen2.5vl:3b docker compose -f src/docker-compose.yml \
                                      -f deployment/llm/docker-compose.ollama.yaml up
```

Pull extra models into a running stack with `./pull_models.sh llava:7b qwen2.5vl:3b`.

### Keeping Ollama on the host instead

`src/docker-compose.yml` alone points the adapter at `host.docker.internal:11434`.
Use that if you already run `ollama serve` on the host and don't want a second copy
of the weights.

## vLLM (needs a GPU)

```bash
VLLM_MODEL=llava-hf/llava-1.5-7b-hf \
docker compose -f src/docker-compose.yml \
               -f deployment/llm/docker-compose.vllm.yaml up
```

**This will not start on a CPU-only host.** vLLM requires an NVIDIA GPU and
`nvidia-container-toolkit`, and a 7B vision model wants roughly 16 GB of VRAM at
the default `--gpu-memory-utilization 0.85`. Use the Ollama overlay on edge or
laptop hardware.

`VLLM_MODEL` is a HuggingFace repo id (`llava-hf/llava-1.5-7b-hf`), not an Ollama
tag. `LLM_MODEL` is the name vLLM serves it under and the name the adapter asks
for — the overlay keeps the two in sync via `--served-model-name`. First start
downloads weights and can take several minutes; the healthcheck allows 5 minutes
before the adapter gives up waiting.

## Registering the backend with the ensemble

The ensemble no longer bakes its member list into the image. It reads
`INFERENCE_BACKENDS` from the environment, as comma-separated `name=kind=url`:

```yaml
ensemble:
  environment:
    INFERENCE_BACKENDS: >-
      mobilenetv2=cnn=http://mobilenetv2:5012/inference,
      efficientnetb0=cnn=http://efficientnetb0:5012/inference,
      llava=llm=http://llava-service:5012/inference
```

`kind` decides the payload, and is stated rather than guessed from the hostname:

- `cnn` — the raw 224×224×3 RGB buffer that preprocessing produces.
- `llm` — the same image JPEG-encoded, since vision models want a real image file.

Getting `kind` wrong sends a raw pixel buffer to a model expecting a JPEG, which
fails at the engine rather than anywhere obvious — so name it correctly when
adding a backend.

If `INFERENCE_BACKENDS` is unset, the ensemble falls back to
`src/ensemble/ensemble_service.yaml`, which keeps the old behaviour (bare model
names, URLs derived from `DOCKER`/`OPENZITI`). Existing k8s manifests are unaffected.

## Adapter environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_BACKEND` | `ollama` | `ollama` or `openai`. Anything else fails at startup. |
| `LLM_BASE_URL` | `http://127.0.0.1:11434` | Engine root URL. Falls back to `OLLAMA_HOST` if unset. |
| `LLM_MODEL` | `llava:7b` | Model tag / served name to request. |
| `LLM_PROMPT` | `What is in the image?` | Prompt sent with every image. |
| `LLM_API_KEY` | unset | Sent as `Authorization: Bearer` on the `openai` path. |
| `MODEL_CALL_TIMEOUT` | `60` | Seconds before the call is abandoned (504). |
| `MAX_IMAGE_SIZE` | `10485760` | Reject larger payloads with 413. |
| `SAVE_DEBUG_IMAGE` | `false` | Write each incoming image to `/tmp` for inspection. |

`GET /health` reports the resolved backend, model, and base URL — the quickest way
to confirm the adapter is pointed where you think it is.

## When a backend is down

The ensemble does not fail the whole request if one member fails. Each entry in
`results` carries its own status, and failures are listed by name:

```json
{
  "results": [
    {"backend": "mobilenetv2", "kind": "cnn", "status": "ok", "result": {...}},
    {"backend": "llava", "kind": "llm", "status": "error",
     "error": "ClientConnectorError: Cannot connect to host llava-service:5012"}
  ],
  "failed": ["llava"],
  "request_id": "...",
  "timestamp": "..."
}
```

Previously a dead backend just produced a shorter results list with nothing to say
it had ever been asked.
