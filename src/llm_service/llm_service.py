# llm_service.py
import os
import base64
import hashlib
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from typing import Optional, Tuple
from ollama import AsyncClient

# from email.parser import BytesParser
# from email.policy import default as email_policy
import re

chosen_model = os.environ["CHOSEN_MODEL"]
if os.environ.get("MANUAL_TRACING"):
    span_processor_endpoint = os.environ.get("OTEL_ENDPOINT")
    if span_processor_endpoint is None:
        raise Exception("Manual debugging requires OTEL_ENDPOINT environment variable")

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor

    # from opentelemetry.sdk.metrics import MeterProvider
    # from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    RequestsInstrumentor().instrument()
    LangchainInstrumentor().instrument()

    # Service name is required for most backends
    resource = Resource(
        attributes={SERVICE_NAME: f"LLM inference-{chosen_model.lower()}"}
    )

    trace_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=span_processor_endpoint))
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)

    tracer = trace.get_tracer(__name__)

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# read env with sensible defaults
ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
if not ollama_host.startswith("http://") and not ollama_host.startswith("https://"):
    ollama_host = "http://" + ollama_host
LLM_MODEL = os.getenv("LLM_MODEL", "llava:7b")

# create a single AsyncClient for reuse
client = AsyncClient(host=ollama_host)


@app.on_event("shutdown")
async def _shutdown():
    try:
        await client.aclose()
    except Exception:
        pass


async def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


async def _manual_parse_multipart(
    body: bytes, content_type_header: str
) -> Optional[Tuple[bytes, str]]:
    """
    Heuristic fallback: try to extract the first file part from a multipart body
    even if request.form() failed. Returns (file_bytes, content_type) or None.
    """
    try:
        # try to extract boundary from content-type header
        m = re.search(
            r"boundary=([^;]+)", content_type_header or "", flags=re.IGNORECASE
        )
        boundary = None
        if m:
            boundary = m.group(1).strip().strip('"')
        else:
            # guess boundary from body start
            if body.startswith(b"--"):
                first_line = body.split(b"\r\n", 1)[0]
                # first_line looks like b'--<boundary>'
                boundary = first_line.decode(errors="ignore").lstrip("--")

        if not boundary:
            return None

        marker = b"--" + boundary.encode()
        parts = body.split(marker)
        for part in parts:
            if not part or b"Content-Disposition:" not in part:
                continue
            # locate start of payload after double CRLF
            sep = b"\r\n\r\n"
            idx = part.find(sep)
            if idx == -1:
                continue
            payload = part[idx + len(sep) :]
            # strip possible trailing boundary markers
            payload = payload.rstrip(b"\r\n--")
            # find content type in part
            ct_match = re.search(
                rb"Content-Type:\s*([^\r\n]+)", part, flags=re.IGNORECASE
            )
            ct = (
                ct_match.group(1).decode().strip()
                if ct_match
                else "application/octet-stream"
            )
            # sanity: ensure payload looks non-empty
            if len(payload) > 0:
                return payload, ct
    except Exception as e:
        logging.debug("manual multipart parse error: %s", e)
    return None


async def _extract_image_bytes(
    request: Request, file: Optional[UploadFile] = None
) -> Tuple[bytes, str]:
    """
    Robust extractor: returns (image_bytes, content_type)
    Supports:
      - FastAPI UploadFile (multipart)
      - request.form() multipart parsing
      - raw binary body (application/octet-stream or image/*)
      - JSON {"image": "<base64>"} or JSON body that is a base64 string
      - fallback: manual multipart parsing heuristics (to support ensemble forwarding)
    """
    # 1) If FastAPI already provided an UploadFile (multipart form with declared file param)
    if file is not None:
        data = await file.read()
        ct = file.content_type or "application/octet-stream"
        return data, ct

    content_type = (request.headers.get("content-type") or "").lower()
    body = await request.body()  # read raw payload once

    # 2) If the incoming content is multipart, prefer request.form() parsing
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            # find first UploadFile-like object or a field that decodes to base64
            for v in form.values():
                # UploadFile and starlette.datastructures.UploadFile have .file / .read
                if hasattr(v, "read"):
                    try:
                        file_bytes = await v.read()
                        ct = (
                            getattr(v, "content_type", content_type)
                            or "application/octet-stream"
                        )
                        if file_bytes:
                            return file_bytes, ct
                    except Exception:
                        continue
                # string fields may carry base64-encoded image
                if isinstance(v, str):
                    try:
                        b = base64.b64decode(v)
                        return b, "application/octet-stream"
                    except Exception:
                        continue
        except Exception as e:
            logging.debug("request.form() failed or empty: %s", e)

        # 3) fallback to manual parsing if form() didn't yield file
        parsed = await _manual_parse_multipart(body, content_type)
        if parsed:
            return parsed

    # 4) JSON with base64 payload
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        # payload could be dict or string
        if isinstance(payload, str):
            # assume base64 string
            try:
                return base64.b64decode(payload), "application/octet-stream"
            except Exception:
                raise HTTPException(
                    status_code=400, detail="JSON body is string but not valid base64"
                )
        elif isinstance(payload, dict):
            img_b64 = (
                payload.get("image") or payload.get("img") or payload.get("image_b64")
            )
            if not img_b64:
                # maybe nested message with 'file' content
                raise HTTPException(
                    status_code=400, detail="JSON must contain base64 'image' field"
                )
            try:
                return base64.b64decode(img_b64), "application/octet-stream"
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid base64 image in JSON"
                )

    # 5) raw binary or image content: accept body as-is if it looks like an image
    if body:
        # quick magic bytes detection
        if body.startswith(b"\xff\xd8"):  # JPEG
            return body, "image/jpeg"
        if body.startswith(b"\x89PNG"):  # PNG
            return body, "image/png"
        if body[:4] == b"RIFF":  # WEBP/AVI RIFF wrapper
            return body, "image/webp"
        # accept application/octet-stream or unknown content-types as raw bytes
        if (
            "application/octet-stream" in content_type
            or content_type.startswith("image/")
            or content_type == ""
        ):
            # still attempt to detect image headers and return a guess
            guessed = "application/octet-stream"
            if body.startswith(b"\xff\xd8"):
                guessed = "image/jpeg"
            elif body.startswith(b"\x89PNG"):
                guessed = "image/png"
            return body, guessed

    raise HTTPException(
        status_code=400, detail="Unable to extract image bytes from request"
    )


@app.post("/inference")
async def inference(request: Request, file: Optional[UploadFile] = File(None)):
    """
    Accepts:
      - multipart form (field 'file' or any file field)
      - raw binary body (curl --data-binary @image.jpg)
      - JSON {"image": "<base64>"}
      - also works if the body is a raw multipart envelope (ensemble forwarding)
    """
    image_bytes, content_type = await _extract_image_bytes(request, file)

    # debug: save a local copy so you can compare with what ensemble forwarded
    sha = await _sha256_hex(image_bytes)
    tmp_path = f"/tmp/llm_in_{sha[:8]}.bin"
    try:
        with open(tmp_path, "wb") as wf:
            wf.write(image_bytes)
        logging.info(
            "Saved incoming bytes to %s size=%d ct=%s",
            tmp_path,
            len(image_bytes),
            content_type,
        )
    except Exception as e:
        logging.warning("Failed to write debug file: %s", e)

    # Build message-level format (same as your working chat example)
    messages = [
        {
            "role": "user",
            "content": "What is in the image?",
            "images": [image_bytes],
        }
    ]

    # Prefer client's chat() if available, otherwise pass messages to generate()
    try:
        # many AsyncClient implementations provide chat()
        response = await client.chat(model=LLM_MODEL, messages=messages, stream=False)
    except AttributeError:
        # fallback: some clients expose generate(messages=...)
        response = await client.generate(
            model=LLM_MODEL, messages=messages, stream=False
        )
    except Exception as e:
        logging.exception("Ollama client error:")
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

    # Extract text content robustly (handle a few common response shapes)
    content = None
    try:
        if hasattr(response, "message") and getattr(response.message, "content", None):
            content = response.message.content
        elif isinstance(response, dict):
            if (
                "message" in response
                and isinstance(response["message"], dict)
                and "content" in response["message"]
            ):
                content = response["message"]["content"]
            elif "response" in response:
                content = response["response"]
            elif (
                "choices" in response
                and isinstance(response["choices"], list)
                and len(response["choices"]) > 0
            ):
                first = response["choices"][0]
                if isinstance(first, dict):
                    if (
                        "message" in first
                        and isinstance(first["message"], dict)
                        and "content" in first["message"]
                    ):
                        content = first["message"]["content"]
                    elif "text" in first:
                        content = first["text"]
                    else:
                        content = str(first)
                else:
                    content = str(first)
        else:
            content = str(response)
    except Exception:
        content = str(response)

    return {"response": content, "sha256": sha}


# import os
#
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request
# from ollama import AsyncClient
#
# load_dotenv()
#
# app = FastAPI()
#
# ollama_host = os.getenv("OLLAMA_HOST")
# llm_model = os.getenv("LLM_MODEL")
#
#
# @app.post("/inference")
# async def inference(request: Request):
#     image_bytes = await request.body()
#
#     client = AsyncClient(host=ollama_host)
#     response = await client.generate(
#         model=llm_model,
#         prompt="what is in the image?",
#         images=[image_bytes],
#         stream=False,
#     )
#
#     return {"response": response["response"]}
