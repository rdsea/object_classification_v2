from __future__ import annotations

import json
import socket

import yaml


def setup_otel(service_name: str):
    """
    Configure OpenTelemetry logging and tracing for the given service.
    - Always enables console logging.
    - Optionally sends logs and traces to an OTEL collector if OTEL_ENDPOINT is set.
    - Tracing only activates if MANUAL_TRACING is set (for explicit control).
    """

    import logging
    import os
    import platform

    from opentelemetry import _logs, trace
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # --- Basic environment and resource setup ---
    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, env_level, logging.INFO)
    instance = platform.node() or "unknown-instance"

    if env_level not in logging._nameToLevel:
        print(f"[setup_otel] Invalid LOG_LEVEL='{env_level}', defaulting to INFO")

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "service.instance.id": instance,
        }
    )

    # --- Configure console logging ---
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    root_logger = logging.getLogger()

    # --- OpenTelemetry Logging Setup ---
    if otel_endpoint:
        provider = LoggerProvider(resource=resource)
        exporter = OTLPLogExporter(endpoint=otel_endpoint)
        processor = BatchLogRecordProcessor(exporter)
        provider.add_log_record_processor(processor)
        _logs.set_logger_provider(provider)

        otel_handler = LoggingHandler(level=log_level, logger_provider=provider)
        root_logger.addHandler(otel_handler)
        root_logger.setLevel(log_level)

        logging.getLogger(__name__).info(
            f"✅ OTEL logging enabled — service='{service_name}', instance='{instance}', "
            f"endpoint='{otel_endpoint}', log_level='{env_level}'"
        )
    else:
        logging.getLogger(__name__).info(
            "⚠️ OTEL_ENDPOINT not set — logs will only go to console."
        )

    if os.environ.get("MANUAL_TRACING"):
        if not otel_endpoint:
            raise RuntimeError(
                "Manual tracing requires OTEL_ENDPOINT environment variable."
            )

        trace_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
        span_processor = BatchSpanProcessor(span_exporter)
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)

        # tracer = trace.get_tracer(__name__)
        logging.getLogger(__name__).info("OTEL tracing enabled (manual mode).")
    else:
        logging.getLogger(__name__).info(
            "MANUAL_TRACING not set — tracing is disabled."
        )

    return root_logger, trace.get_tracer(__name__)


def get_local_ip():
    try:
        # The following line creates a socket to connect to an external site
        # The IP address returned is the one of the network interface used for the connection
        # '8.8.8.8' is used here as it's a public DNS server by Google
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "Unable to get IP: " + str(e)


def load_config(file_path: str) -> dict | None:
    """
    file_path: file path to load config
    """
    try:
        if "json" in file_path:
            with open(file_path) as f:
                return json.load(f)
        if ("yaml" in file_path) or ("yml" in file_path):
            with open(file_path) as f:
                return yaml.safe_load(f)
        else:
            return None
    except yaml.YAMLError as exc:
        print(exc)
