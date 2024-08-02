import asyncio
import logging
import signal
import sys
from multiprocessing import Process
from typing import Annotated

from executor import ProcessingServiceExecutor
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from rohe.common import rohe_utils
from rohe.service_registry.consul import ConsulClient
from rohe.storage.minio import MinioConnector

# WARN: Is this bad? What if we have multiple instance, do we have to route update request to all to change?
config_lock = asyncio.Lock()  # Lock to control access to the global variable

PORT = 5012
try:
    port = PORT
    config_file = "processing_config.yaml"
    config = rohe_utils.load_config(file_path=config_file)
except Exception as e:
    logging.error(f"Error loading config file: {e}")
    sys.exit(1)
assert config is not None
logging.info(f"Image Processing configuration: {config}")

# Init Minio Connector for uploading image
minio_connector = MinioConnector(config["external_services"]["minio_storage"])

# consul for service register
# register service
local_ip = rohe_utils.get_local_ip()
consul_client = ConsulClient(
    config=config["external_services"]["service_registry"]["consul_config"]
)
service_id = consul_client.service_register(
    name="processing", address=local_ip, tag=["nii_case"], port=port
)


retry_delay: dict = config["processing"]["request"]["retry_delay"]
for k, v in retry_delay.items():
    retry_delay[k] = rohe_utils.parse_time(time_str=v)
logging.info(f"This is parsed time: {config['processing']['request']['retry_delay']}")

executor = ProcessingServiceExecutor(config, minio_connector, consul_client)
executor_process = Process(target=executor.run)
executor_process.start()

app = FastAPI()


# Create a POST endpoint
@app.post("/change_ensemble/")
async def change_ensemble(ensemble_name: str = Form("ensemble_name")):
    try:
        async with config_lock:
            executor.config["external_services"]["inference_service"][
                "ensemble_name"
            ] = ensemble_name
            response = f"Change ensemble to: {ensemble_name} successfully"
            return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


@app.post("/change_config/")
async def change_requirement(configuration: Annotated[dict, Form()]):
    try:
        async with config_lock:
            executor.config = configuration
            response = f"Change ensemble to: {configuration} successfully"
            return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


@app.get("/test/")
async def get_test():
    """
    return message to client to notify them that they are accessing the correct ingestion server
    """
    response = "This is processing service controller"
    return JSONResponse(content={"response": response}, status_code=200)


def signal_handler(sig, frame):
    logging.info("You pressed Ctrl+C! Gracefully shutting down.")
    consul_client.service_deregister(id=service_id)
    executor_process.kill()
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)
