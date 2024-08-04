import logging
import os
import signal
import sys

import numpy as np
from fastapi import FastAPI, Request
from object_classification_agent import ObjectClassificationAgent
from rohe.common import rohe_utils
from rohe.service_registry.consul import ConsulClient
from rohe.storage.minio import MinioConnector

PORT = int(os.environ["PORT"])
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
# chosen_model_id = config["model_info"]["chosen_model_id"]
chosen_model_id = os.environ["CHOSEN_MODEL"]
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
async def inference(request: Request):
    image_bytes = await request.body()
    reconstructed_image = np.frombuffer(image_bytes, dtype=np.uint8)
    reconstructed_image = reconstructed_image.reshape((32, 32, 3))
    return ml_agent.predict(reconstructed_image)


def signal_handler(sig, frame):
    print("You pressed Ctrl+C! Gracefully shutting down.")
    consul_client.service_deregister(id=service_id)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
