import asyncio
import logging
import os
import sys
import time
from typing import Optional
from uuid import uuid4

import aiohttp
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from image_processing_functions import resize


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def is_env_true(var_name: str, default: str = "false") -> bool:
    value = os.environ.get(var_name, default).strip().lower()
    return value in {"true", "1", "yes", "on", "t"}


class Settings:
    def __init__(self) -> None:
        self.openziti_enabled: bool = is_env_true("OPENZITI")
        self.manual_tracing: bool = is_env_true("MANUAL_TRACING")
        self.send_to_queue: bool = is_env_true("SEND_TO_QUEUE")
        self.connect_url: str = os.environ.get(
            "CONNECT_URL",
            "http://localhost:5011/ensemble_service",
        ).strip()
        self.otel_endpoint: Optional[str] = os.environ.get("OTEL_ENDPOINT")
        self.request_timeout_seconds: float = float(
            os.environ.get("REQUEST_TIMEOUT_SECONDS", "10")
        )

        if not self.connect_url:
            raise RuntimeError("CONNECT_URL is empty or not set")

        if self.manual_tracing and not self.otel_endpoint:
            raise RuntimeError("MANUAL_TRACING is enabled but OTEL_ENDPOINT is not set")

    def log_summary(self) -> None:
        logging.info(
            "Service configuration: openziti_enabled=%s, manual_tracing=%s, connect_url=%s, send_to_queue=%s, timeout=%s",
            self.openziti_enabled,
            self.manual_tracing,
            self.connect_url,
            self.send_to_queue,
            self.request_timeout_seconds,
        )


settings = Settings()
settings.log_summary()


# Load util path and config
current_directory = os.path.dirname(os.path.abspath(__file__))
util_directory = os.path.join(current_directory, "..", "util")
if util_directory not in sys.path:
    sys.path.append(util_directory)

import utils  # noqa: E402


try:
    config_file = "preprocessing_config.yaml"
    config = utils.load_config(file_path=config_file)
except Exception as e:
    logging.error("Error loading config file: %s", e)
    sys.exit(1)

if config is None:
    logging.error("Config file loaded as None")
    sys.exit(1)


# Optional tracing setup
if settings.manual_tracing:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    AioHttpClientInstrumentor().instrument()

    resource = Resource(attributes={SERVICE_NAME: "preprocessing"})
    trace_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)
    tracer = trace.get_tracer(__name__)


accepted_file_types = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/heic",
    "image/heif",
    "image/heics",
    "png",
    "jpeg",
    "jpg",
    "heic",
    "heif",
    "heics",
}


app = FastAPI(title="preprocessing-service")


def validate_image_type(content_type: Optional[str]) -> None:
    if content_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="UploadFile has no content type",
        )

    if content_type not in accepted_file_types:
        logging.info("Rejected content type: %s", content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"UploadFile with type {content_type} is not accepted. "
                f"Accepted types: {sorted(accepted_file_types)}"
            ),
        )


def extract_timestamp_header(request: Request) -> str:
    timestamp = request.headers.get("Timestamp")
    if not timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required Timestamp header",
        )
    return timestamp


def decode_image(contents: bytes) -> np.ndarray:
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unsupported image data",
        )

    # OpenCV decodes as BGR; convert to RGB for downstream processing.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if image.shape != (224, 224, 3):
        return resize(image)
    return image


@app.get("/test/")
async def get_test():
    response = {
        "response": "This is processing service controller",
        "openziti_enabled": settings.openziti_enabled,
        "connect_url": settings.connect_url,
    }
    return JSONResponse(content=response, status_code=200)


@app.post("/preprocessing")
async def processing_image(file: UploadFile, request: Request):
    logging.info("Incoming request headers: %s", dict(request.headers))

    validate_image_type(file.content_type)
    timestamp = extract_timestamp_header(request)

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    image = decode_image(contents)

    start_processing = time.time()
    processed_image = preprocess_image(image)
    processing_time_ms = (time.time() - start_processing) * 1000.0

    logging.info("Image preprocessing took %.2f ms", processing_time_ms)

    image_bytes = processed_image.tobytes()
    request_id = str(uuid4())

    headers = {
        "Timestamp": timestamp,
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(image_bytes)),
    }

    try:
        timeout = aiohttp.ClientTimeout(total=settings.request_timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            logging.info(
                "Sending request_id=%s to %s", request_id, settings.connect_url
            )

            async with session.post(
                url=settings.connect_url,
                headers=headers,
                data=image_bytes,
                params={"request_id": request_id},
            ) as response:
                response_text = await response.text()

                if response.status != 200:
                    logging.error(
                        "Downstream service returned status=%s body=%s",
                        response.status,
                        response_text,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=(
                            "Failed to send image to downstream service. "
                            f"Status code: {response.status}"
                        ),
                    )

        return JSONResponse(
            content={
                "message": "File accepted",
                "request_id": request_id,
                "processing_time_ms": round(processing_time_ms, 2),
            },
            status_code=status.HTTP_200_OK,
        )

    except aiohttp.ClientError as e:
        logging.error("Client error while connecting to downstream service: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to connect to downstream service",
        )

    except asyncio.TimeoutError:
        logging.error("Request to downstream service timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request to downstream service timed out",
        )

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("Unexpected error during preprocessing request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )


if settings.manual_tracing:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
