from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import aio_pika
import aiohttp
import ensemble_function
from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import JSONResponse

from util.utils import load_config, setup_otel

SERVICE_NAME = os.environ.get("SERVICE_NAME", "ensemble")

SEND_TO_QUEUE = os.environ.get("SEND_TO_QUEUE", "false").lower() == "true"

setup_otel(SERVICE_NAME)

config_lock = asyncio.Lock()  # Lock to control access to the global variable


config_file = "ensemble_service.yaml"
config = load_config(file_path=config_file)

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
    async with session.post(url, data=image_data, headers=headers) as response:
        return await response.json()  # Assuming the response is JSON


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

    async with aiohttp.ClientSession(trust_env=True) as session:
        tasks = [
            asyncio.create_task(send_post_request(session, url, image_data, headers))
            for url in INFERENCE_SERVICE_URLS
        ]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        results = []
        for task in done:
            results.append(await task)

        # Run ensemble function on the results
        final_result = chosen_ensemble_function(results, request_id)
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
    from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    AioHttpClientInstrumentor().instrument()
    AioPikaInstrumentor().instrument()
    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
