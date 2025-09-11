from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import urlparse

import aio_pika
import aiohttp
import ensemble_function
import numpy as np
import cv2
from fastapi import BackgroundTasks, FastAPI, Form, Request, HTTPException
from fastapi.responses import JSONResponse

if os.environ.get("MANUAL_TRACING"):
    span_processor_endpoint = os.environ.get("OTEL_ENDPOINT")
    if span_processor_endpoint is None:
        raise Exception("Manual debugging requires OTEL_ENDPOINT environment variable")

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

    # from opentelemetry.sdk.metrics import MeterProvider
    # from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    AioHttpClientInstrumentor().instrument()

    AioPikaInstrumentor().instrument()
    # Service name is required for most backends
    resource = Resource(attributes={SERVICE_NAME: "ensemble"})

    trace_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=span_processor_endpoint))
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)

    tracer = trace.get_tracer(__name__)

SEND_TO_QUEUE = os.environ.get("SEND_TO_QUEUE", "false").lower() == "true"

current_directory = os.path.dirname(os.path.abspath(__file__))
util_directory = os.path.join(current_directory, "..", "util")
sys.path.append(util_directory)

# TODO: change utils to package that other service can reuse
import utils  # noqa: E402

config_lock = asyncio.Lock()  # Lock to control access to the global variable


config_file = "ensemble_service.yaml"
config = utils.load_config(file_path=config_file)

assert config is not None
logging.debug(f"Ensemble Service configuration: {config}")


def get_inference_service_url(ensemble_chosen: list[str]):
    return [f"http://{item.lower()}-service:5012/inference" for item in ensemble_chosen]


def get_inference_service_url_docker(ensemble_chosen: list[str]):
    return [f"http://{item.lower()}:5012/inference" for item in ensemble_chosen]


def get_inference_service_url_openziti(ensemble_chosen: list[str]):
    return [
        f"http://{item.lower()}.miniziti.private:5012/inference"
        for item in ensemble_chosen
    ]


def get_rabbitmq_connection_url():
    rabbitmq_url = os.environ.get("RABBITMQ_URL")
    username = os.environ.get("RABBITMQ_USERNAME")
    password = os.environ.get("RABBITMQ_PASSWORD")
    if not all([rabbitmq_url, username, password]):
        raise ValueError("RabbitMQ environment variables are not set")
    return f"amqp://{username}:{password}@{rabbitmq_url}"


INFERENCE_SERVICE_URLS = get_inference_service_url(config["ensemble"])
if os.environ.get("DOCKER"):
    INFERENCE_SERVICE_URLS = get_inference_service_url_docker(config["ensemble"])

if os.environ.get("OPENZITI"):
    INFERENCE_SERVICE_URLS = get_inference_service_url_openziti(config["ensemble"])


def get_first_part_of_host(url: str) -> str:
    """
    Extract the first part before the first '-' from the host portion of the URL.
    Examples:
      - http://llm-llava-service:5012/inference  -> "llm"
      - llm-llava-service:5012/inference          -> "llm"
      - http://llm.example.com                    -> "llm"
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        # fallback for weird strings without scheme
        host = url.split("/", 1)[0].split(":", 1)[0]
    return host.split("-", 1)[0].lower()


def is_jpeg_bytes(b: bytes) -> bool:
    return isinstance(b, (bytes, bytearray)) and len(b) >= 2 and b[:2] == b"\xff\xd8"


def ensure_jpeg_bytes(image_data) -> bytes:
    """
    Return JPEG-encoded bytes.
    Accepts:
      - bytes (if already JPEG, return as-is; otherwise try to decode and re-encode)
      - numpy array (RGB or BGR) -> convert to BGR then JPEG encode
    Raises HTTPException on failure.
    """
    try:
        # already bytes
        if isinstance(image_data, (bytes, bytearray)):
            if is_jpeg_bytes(image_data):
                return bytes(image_data)
            # try decode (may be PNG etc.) and re-encode
            arr = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                raise ValueError("Could not decode provided image bytes")
            success, buf = cv2.imencode(".jpg", img)
            if not success:
                raise ValueError("cv2.imencode failed")
            return buf.tobytes()

        # assume numpy array
        img = image_data
        # convert float/other dtypes to uint8 if needed
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        # assume RGB input, convert to BGR for OpenCV
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = img  # maybe already BGR
        success, buf = cv2.imencode(".jpg", bgr)
        if not success:
            raise ValueError("cv2.imencode failed")
        return buf.tobytes()
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to ensure JPEG bytes")
        raise HTTPException(status_code=500, detail=f"Failed to prepare JPEG: {exc}")


def encode_to_jpeg_bytes(
    image_data: bytes, height: int = 224, width: int = 224
) -> bytes:
    """
    Convert raw RGB bytes (tobytes()) into JPEG bytes suitable for LLM endpoints.
    """
    try:
        # reshape back to (H, W, 3)
        arr = np.frombuffer(image_data, dtype=np.uint8)
        img = arr.reshape((height, width, 3))  # must match preprocessing output shape

        # encode as JPEG
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("cv2.imencode failed")

        return buffer.tobytes()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to encode raw image bytes to JPEG: {exc}"
        )


def decode_jpeg_to_rgb_array(
    jpeg_bytes: bytes,
) -> tuple[bytes, tuple[int, int, int], str]:
    """
    Decode JPEG bytes to RGB numpy array and return (raw_bytes, shape, dtype_str).
    Useful if a downstream expects raw pixel buffer.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image bytes")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.tobytes(), img_rgb.shape, str(img_rgb.dtype)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SEND_TO_QUEUE:
        rabbitmq_url = get_rabbitmq_connection_url()
        connection = await aio_pika.connect_robust(rabbitmq_url)
        channel = await connection.channel()
        queue_name = os.environ.get("RABBITMQ_QUEUE_NAME")
        if not queue_name:
            raise ValueError("RABBITMQ_QUEUE_NAME environment variable is not set")
        await channel.declare_queue(queue_name, durable=True)

        app.state.rabbitmq_connection = connection
        app.state.rabbitmq_channel = channel

        yield  # Application runs during this period

        # Close RabbitMQ connection and channel at shutdown
        await channel.close()
        await connection.close()
    else:
        yield


app = FastAPI(lifespan=lifespan)
app.state.config = config


async def send_post_request(
    session: aiohttp.ClientSession, url: str, image_data: bytes, headers
):
    # image_data is raw bytes (JPEG/PNG or already JPEG)
    async with session.post(url, data=image_data, headers=headers) as response:
        return await response.json()  # Assuming the inference service returns JSON


async def process_image_task(
    image_data: bytes, request_id: str, headers, timestamp: str
):
    chosen_ensemble_function = getattr(
        ensemble_function,
        app.state.config["aggregating"]["aggregating_func"]["func_name"],
    )
    logging.info(f"List service url: {INFERENCE_SERVICE_URLS}")

    if not INFERENCE_SERVICE_URLS:
        raise RuntimeError("No inference service url")

    # Precompute JPEG bytes if any service requires `llm` prefix
    needs_llm = any(get_first_part_of_host(u) == "llm" for u in INFERENCE_SERVICE_URLS)
    llm_image_bytes = None
    if needs_llm:
        # encode once and reuse
        llm_image_bytes = encode_to_jpeg_bytes(image_data)

    async with aiohttp.ClientSession(trust_env=True) as session:
        tasks = []
        for url in INFERENCE_SERVICE_URLS:
            first_part = get_first_part_of_host(url)
            payload = llm_image_bytes if first_part == "llm" else image_data

            # Prepare headers to forward, but strip/override headers that would be incorrect
            forward_headers = {
                k: v
                for k, v in headers.items()
                if k.lower() not in ("content-length", "host")
            }
            if first_part == "llm":
                # ensure the content-type matches JPEG
                forward_headers["Content-Type"] = "image/jpeg"

            tasks.append(
                asyncio.create_task(
                    send_post_request(session, url, payload, forward_headers)
                )
            )

        # gather results, but don't fail all if one fails — log and continue
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                logging.exception(f"Call to {INFERENCE_SERVICE_URLS[idx]} failed: {r}")
            else:
                processed_results.append(r)

        # Run ensemble function on the results
        final_result = chosen_ensemble_function(processed_results, request_id)
        final_result["Timestamp"] = timestamp
        logging.info(f"Ensembled result: {final_result}")

        if SEND_TO_QUEUE:
            channel = app.state.rabbitmq_channel
            queue_name = os.environ.get("RABBITMQ_QUEUE_NAME")
            if not queue_name:
                raise ValueError("RABBITMQ_QUEUE_NAME environment variable is not set")
            message_body = json.dumps(final_result).encode()
            message = aio_pika.Message(body=message_body)
            await channel.default_exchange.publish(message, routing_key=queue_name)
            logging.info(f"Sent result to RabbitMQ queue {queue_name}")


@app.post("/ensemble_service")
async def ensemble(
    request: Request,
    background_tasks: BackgroundTasks,
):
    try:
        image_bytes = await request.body()
        request_id = request.query_params["request_id"]
        headers = request.headers
        # logging.info(image_bytes)
        background_tasks.add_task(
            process_image_task, image_bytes, request_id, headers, headers["Timestamp"]
        )

        response = "Success to add image to Ensemble Service"
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.exception(f"Error: {e}")
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


@app.post("/change_config")
async def change_requirement(configuration: Annotated[dict, Form()]):
    try:
        async with config_lock:
            app.state.config = configuration
            global INFERENCE_SERVICE_URLS
            INFERENCE_SERVICE_URLS = get_inference_service_url(
                app.state.config["ensemble"]
            )
            response = f"Change ensemble to: {configuration} successfully"
            return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.exception(f"Error: {e}")
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


if os.environ.get("MANUAL_TRACING"):
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
# import asyncio
# import json
# import logging
# import os
# import sys
# from contextlib import asynccontextmanager
# from typing import Annotated
#
# import aio_pika
# import aiohttp
# import ensemble_function
# from fastapi import BackgroundTasks, FastAPI, Form, Request
# from fastapi.responses import JSONResponse
#
# if os.environ.get("MANUAL_TRACING"):
#     span_processor_endpoint = os.environ.get("OTEL_ENDPOINT")
#     if span_processor_endpoint is None:
#         raise Exception("Manual debugging requires OTEL_ENDPOINT environment variable")
#
#     from opentelemetry import trace
#     from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
#     from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
#     from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
#
#     # from opentelemetry.sdk.metrics import MeterProvider
#     # from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
#     from opentelemetry.sdk.resources import SERVICE_NAME, Resource
#     from opentelemetry.sdk.trace import TracerProvider
#     from opentelemetry.sdk.trace.export import BatchSpanProcessor
#
#     AioHttpClientInstrumentor().instrument()
#
#     AioPikaInstrumentor().instrument()
#     # Service name is required for most backends
#     resource = Resource(attributes={SERVICE_NAME: "ensemble"})
#
#     trace_provider = TracerProvider(resource=resource)
#     processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=span_processor_endpoint))
#     trace_provider.add_span_processor(processor)
#     trace.set_tracer_provider(trace_provider)
#
#     tracer = trace.get_tracer(__name__)
#
# SEND_TO_QUEUE = os.environ.get("SEND_TO_QUEUE", "false").lower() == "true"
#
# current_directory = os.path.dirname(os.path.abspath(__file__))
# util_directory = os.path.join(current_directory, "..", "util")
# sys.path.append(util_directory)
#
# # TODO: change utils to package that other service can reuse
# import utils  # noqa: E402
#
# config_lock = asyncio.Lock()  # Lock to control access to the global variable
#
#
# config_file = "ensemble_service.yaml"
# config = utils.load_config(file_path=config_file)
#
# assert config is not None
# logging.debug(f"Ensemble Service configuration: {config}")
#
#
# def get_inference_service_url(ensemble_chosen: list[str]):
#     return [f"http://{item.lower()}-service:5012/inference" for item in ensemble_chosen]
#
#
# def get_inference_service_url_docker(ensemble_chosen: list[str]):
#     return [f"http://{item.lower()}:5012/inference" for item in ensemble_chosen]
#
#
# def get_inference_service_url_openziti(ensemble_chosen: list[str]):
#     return [
#         f"http://{item.lower()}.miniziti.private:5012/inference"
#         for item in ensemble_chosen
#     ]
#
#
# def get_rabbitmq_connection_url():
#     rabbitmq_url = os.environ.get("RABBITMQ_URL")
#     username = os.environ.get("RABBITMQ_USERNAME")
#     password = os.environ.get("RABBITMQ_PASSWORD")
#     if not all([rabbitmq_url, username, password]):
#         raise ValueError("RabbitMQ environment variables are not set")
#     return f"amqp://{username}:{password}@{rabbitmq_url}"
#
#
# INFERENCE_SERVICE_URLS = get_inference_service_url(config["ensemble"])
# if os.environ.get("DOCKER"):
#     INFERENCE_SERVICE_URLS = get_inference_service_url_docker(config["ensemble"])
#
# if os.environ.get("OPENZITI"):
#     INFERENCE_SERVICE_URLS = get_inference_service_url_openziti(config["ensemble"])
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     if SEND_TO_QUEUE:
#         rabbitmq_url = get_rabbitmq_connection_url()
#         connection = await aio_pika.connect_robust(rabbitmq_url)
#         channel = await connection.channel()
#         queue_name = os.environ.get("RABBITMQ_QUEUE_NAME")
#         if not queue_name:
#             raise ValueError("RABBITMQ_QUEUE_NAME environment variable is not set")
#         await channel.declare_queue(queue_name, durable=True)
#
#         app.state.rabbitmq_connection = connection
#         app.state.rabbitmq_channel = channel
#
#         yield  # Application runs during this period
#
#         # Close RabbitMQ connection and channel at shutdown
#         await channel.close()
#         await connection.close()
#     else:
#         yield
#
#
# app = FastAPI(lifespan=lifespan)
# app.state.config = config
#
#
# async def send_post_request(
#     session: aiohttp.ClientSession, url: str, image_data: bytes, headers
# ):
#     async with session.post(url, data=image_data, headers=headers) as response:
#         return await response.json()  # Assuming the response is JSON
#
#
# async def process_image_task(
#     image_data: bytes, request_id: str, headers, timestamp: str
# ):
#     chosen_ensemble_function = getattr(
#         ensemble_function,
#         app.state.config["aggregating"]["aggregating_func"]["func_name"],
#     )
#     logging.info(f"List service url: {INFERENCE_SERVICE_URLS}")
#
#     if not INFERENCE_SERVICE_URLS:
#         raise RuntimeError("No inference service url")
#
#     async with aiohttp.ClientSession(trust_env=True) as session:
#         tasks = [
#
#             asyncio.create_task(send_post_request(session, url, image_data, headers))
#             for url in INFERENCE_SERVICE_URLS
#
#             #
#             if first_part == "llm":
#                 success, buffer = cv2.imencode(
#                     ".jpg", cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
#                 )
#                 if not success:
#                     raise HTTPException(status_code=500, detail="Failed to encode image")
#                 image_bytes = buffer.tobytes()
#                 asyncio.create_task(send_post_request(session, url, image_bytes, headers))
#                 for url in INFERENCE_SERVICE_URLS
#         ]
#         done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
#
#         results = []
#         for task in done:
#             results.append(await task)
#
#         # Run ensemble function on the results
#         final_result = chosen_ensemble_function(results, request_id)
#         final_result["Timestamp"] = timestamp
#         logging.info(f"Ensembled result: {final_result}")
#
#         if SEND_TO_QUEUE:
#             channel = app.state.rabbitmq_channel
#             queue_name = os.environ.get("RABBITMQ_QUEUE_NAME")
#             if not queue_name:
#                 raise ValueError("RABBITMQ_QUEUE_NAME environment variable is not set")
#             message_body = json.dumps(final_result).encode()
#             message = aio_pika.Message(body=message_body)
#             await channel.default_exchange.publish(message, routing_key=queue_name)
#             logging.info(f"Sent result to RabbitMQ queue {queue_name}")
#
#
# @app.post("/ensemble_service")
# async def ensemble(
#     request: Request,
#     background_tasks: BackgroundTasks,
# ):
#     try:
#         image_bytes = await request.body()
#         request_id = request.query_params["request_id"]
#         headers = request.headers
#         # logging.info(image_bytes)
#         background_tasks.add_task(
#             process_image_task, image_bytes, request_id, headers, headers["Timestamp"]
#         )
#
#         response = "Success to add image to Ensemble Service"
#         return JSONResponse(content={"response": response}, status_code=200)
#     except Exception as e:
#         logging.exception(f"Error: {e}")
#         return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)
#
#
# @app.post("/change_config")
# async def change_requirement(configuration: Annotated[dict, Form()]):
#     try:
#         async with config_lock:
#             app.state.config = configuration
#             global INFERENCE_SERVICE_URLS
#             INFERENCE_SERVICE_URLS = get_inference_service_url(
#                 app.state.config["ensemble"]
#             )
#             response = f"Change ensemble to: {configuration} successfully"
#             return JSONResponse(content={"response": response}, status_code=200)
#     except Exception as e:
#         logging.exception(f"Error: {e}")
#         return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)
#
#
# if os.environ.get("MANUAL_TRACING"):
#     from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
#
#     FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
