import logging
import signal
import sys
from typing import Annotated

from fastapi import FastAPI, Form, UploadFile
from object_classification_agent import ObjectClassificationAgent
from rohe.common import rohe_utils
from rohe.service_registry.consul import ConsulClient
from rohe.storage.minio import MinioConnector

PORT = 11020
try:
    config_file = "inference_service.yaml"
    config = rohe_utils.load_config(file_path=config_file)
except Exception as e:
    logging.error(f"Error loading config file: {e}")
    sys.exit(1)
assert config is not None
logging.info(f"Ingestion configuration: {config}")

local_ip = rohe_utils.get_local_ip()
consul_client = ConsulClient(
    config=config["external_services"]["service_registry"]["consul_config"]
)
chosen_model_id = config["model_info"]["chosen_model_id"]
service_id = consul_client.service_register(
    name=chosen_model_id,
    address=local_ip,
    tag=["nii_case", "vgg", chosen_model_id],
    port=PORT,
)

minio_connector = MinioConnector(config["external_services"]["minio_storage"])


ml_agent = ObjectClassificationAgent(
    load_model_params=config["model_info"]["load_model_params"],
    model_id=chosen_model_id,
    input_shape=config["model_info"]["input_shape"],
)

app = FastAPI()


@app.post("/inference/")
async def inference(
    image: UploadFile,
    timestamp: Annotated[str, Form()],
    device_id: Annotated[str, Form()],
    image_extension: Annotated[str, Form()],
    dtype: Annotated[str, Form()],
    shape: Annotated[str, Form()],
):
    pass


def signal_handler(sig, frame):
    print("You pressed Ctrl+C! Gracefully shutting down.")
    consul_client.service_deregister(id=service_id)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
