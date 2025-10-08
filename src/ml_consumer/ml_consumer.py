import asyncio
import datetime
import json
import logging
import os
import sys
import time
import uuid
from multiprocessing import Process, current_process

import aio_pika
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

from util.utils import setup_logging

MAX_RETRIES = 10
INITIAL_DELAY = 2
MAX_DELAY = 60
NUM_PROCESSES = int(os.getenv("NUM_PROCESSES", "1"))

# RabbitMQ configuration
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME", "default_username")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "default_password")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_URL = f"amqp://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}"
QUEUE_NAME = os.getenv("RABBITMQ_QUEUE_NAME", "object_detection_result")

# ScyllaDB configuration
SCYLLA_USERNAME = os.getenv("SCYLLA_USERNAME", "default_username")
SCYLLA_PASSWORD = os.getenv("SCYLLA_PASSWORD", "default_password")
SCYLLA_HOST = os.getenv("SCYLLA_HOST", "scylla")
SCYLLA_PORT = int(os.getenv("SCYLLA_PORT", 9042))
KEYSPACE = "object_detection"
TABLE_NAME = "results"

setup_logging()

# === TRACING SETUP ===
if os.environ.get("MANUAL_TRACING"):
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    span_processor_endpoint = os.environ.get("OTEL_ENDPOINT")
    if not span_processor_endpoint:
        raise Exception("OTEL_ENDPOINT env variable is required for manual tracing")

    AioPikaInstrumentor().instrument()

    trace_provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: "ml-consumer"})
    )
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=span_processor_endpoint))
    )
    trace.set_tracer_provider(trace_provider)
    tracer = trace.get_tracer(__name__)
else:
    tracer = None


async def process_message(message: aio_pika.IncomingMessage, session):
    def bridge_future(response_future, loop):
        future = loop.create_future()

        def on_success(result):
            loop.call_soon_threadsafe(future.set_result, result)

        def on_error(ex):
            loop.call_soon_threadsafe(future.set_exception, ex)

        response_future.add_callback(on_success)
        response_future.add_errback(on_error)

        return future

    async with message.process():
        try:
            body = message.body.decode()
            data = json.loads(body)
            data["endtime"] = time.time()
            request_id = uuid.UUID(data.get("request_id", str(uuid.uuid4())))
            prediction_result = data["prediction"][0]
            timestamp = data.get("timestamp", time.time())
            dt_object = datetime.datetime.fromtimestamp(float(timestamp))

            query = f"""
                INSERT INTO {KEYSPACE}.{TABLE_NAME} (id, timestamp, prediction, confidence)
                VALUES (%s, %s, %s, %s)
            """
            values = (request_id, dt_object, prediction_result[0], prediction_result[1])

            loop = asyncio.get_running_loop()
            response_future = session.execute_async(SimpleStatement(query), values)

            if tracer:
                with tracer.start_as_current_span("process_message") as span:
                    span.set_attribute("process.name", current_process().name)
                    span.set_attribute(
                        "rabbitmq.message_id", message.message_id or "unknown"
                    )
                    span.set_attribute(
                        "rabbitmq.routing_key", message.routing_key or "unknown"
                    )
                    logging.info(f"{current_process().name} received: {data}")
                    await bridge_future(response_future, loop)
                    logging.info(
                        f"{current_process().name} inserted request_id: {request_id}"
                    )
            else:
                logging.info(f"{current_process().name} received: {data}")
                await bridge_future(response_future, loop)
                logging.info(
                    f"{current_process().name} inserted request_id: {request_id}"
                )

        except Exception as e:
            logging.exception(f"Error: {e}")
            if tracer:
                span.record_exception(e)


async def consume():
    if tracer:
        with tracer.start_as_current_span("consume_worker") as span:
            await _consume_logic(span)
    else:
        await _consume_logic(None)


async def _consume_logic(span):
    cluster = Cluster([SCYLLA_HOST], port=SCYLLA_PORT)
    session = cluster.connect()

    retries = 0
    delay = INITIAL_DELAY

    while retries < MAX_RETRIES:
        try:
            if span:
                span.add_event("Attempting RabbitMQ connection")
            connection = await aio_pika.connect_robust(RABBITMQ_URL)
            break
        except Exception as e:
            logging.warning(f"Connection failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            retries += 1
            delay = min(delay * 2, MAX_DELAY)
    else:
        logging.error("Exceeded max retries to connect to RabbitMQ.")
        sys.exit(1)

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(QUEUE_NAME, durable=True)
        logging.info(f"{current_process().name} consuming from queue: {QUEUE_NAME}")
        await queue.consume(lambda msg: process_message(msg, session))
        await asyncio.Future()  # Keep running


def start_worker():
    asyncio.run(consume())


if __name__ == "__main__":
    processes = [
        Process(target=start_worker, name=f"Worker-{i}") for i in range(NUM_PROCESSES)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
