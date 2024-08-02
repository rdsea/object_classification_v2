import datetime
import random
import signal
import string
import sys
import time
import uuid
from enum import Enum
from typing import Annotated

import requests
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
from rohe.common import rohe_utils
from rohe.common.logger import logger
from rohe.service_registry.consul import ConsulClient
from rohe.storage.minio import MinioConnector

PORT = 5010
try:
    config_file = "ingestion_config.yaml"
    config = rohe_utils.load_config(file_path=config_file)
except Exception as e:
    logger.exception(f"Error loading config file: {e}")
    sys.exit(1)

assert config is not None

logger.info(f"Ingestion configuration: {config}")

# Init Minio Connector for uploading image
minio_connector = MinioConnector(config["external_services"]["minio_storage"])

local_ip = rohe_utils.get_local_ip()
consul_client = ConsulClient(
    config=config["external_services"]["service_registry"]["consul_config"]
)
service_id = consul_client.service_register(
    name="ingestion", address=local_ip, tag=["nii_case"], port=PORT
)


def validate_image_extension(file_extension, supported_extensions: list | None = None):
    if not supported_extensions:
        supported_extensions = ["npy", "jpg", "jpeg", "png", "webp"]
    # print("This is supported extension: ")
    return file_extension.lower() in supported_extensions


def generate_request_id() -> str:
    date_str = rohe_utils.get_current_utc_timestamp()
    uuid_str = str(uuid.uuid4())
    additional_str = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    request_id = f"{date_str}-{uuid_str}-{additional_str}"
    return request_id


app = FastAPI()
app.state.config = config


def get_image_info_service():
    tags = app.state.config["external_services"]["service_registry"]["service"][
        "image_info"
    ]["tags"]
    query_type = app.state.config["external_services"]["service_registry"]["service"][
        "image_info"
    ]["type"]
    try:
        for _ in range(1, 3):
            image_info_service_list = rohe_utils.handle_service_query(
                consul_client=consul_client,
                service_name="image_info",
                query_type=query_type,
                tags=tags,
            )
            if image_info_service_list:
                return image_info_service_list[0]
            time.sleep(1)
            logger.info("Waiting for image info service to be available")
    except Exception as e:
        logger.exception(f"Error: {e}")
        return JSONResponse(
            content={"response": "Error in querying Image Info Service"},
            status_code=400,
        )
    return JSONResponse(
        content={"response": "Image Info Service is not available"},
        status_code=400,
    )


class ImageExtensionEnum(str, Enum):
    npy = "npy"
    jpg = "jph"
    jpeg = "jpeg"
    png = "png"
    webp = "webp"


@app.post("/ingestion_service/")
async def get_image(
    image: UploadFile,
    image_extension: ImageExtensionEnum,
    timestamp: Annotated[datetime.datetime, Form()],
    device_id: Annotated[str, Form()],
):
    try:
        logger.info(
            f"Received request with timestamp: {timestamp}, device_id: {device_id}, image_extension: {image_extension}"
        )
        logger.info(image.filename)

        image_info_service = get_image_info_service()
        if isinstance(image_info_service, JSONResponse):
            return image_info_service
        image_info_endpoint = "image_info_service/add"
        image_info_service_url = f"http://{image_info_service['Address']}:{image_info_service['Port']}/{image_info_endpoint}"
        logger.info(f"This is image info endpoint: {image_info_service_url}")

        try:
            image_content = await image.read()
            request_id = generate_request_id()
            remote_filename = (
                f"{device_id}/{timestamp!s}/{request_id}.{image_extension}"
            )
        except Exception as e:
            logger.exception(f"Error: {e}")
            return JSONResponse(
                content={"response": "Error in init request info"}, status_code=400
            )
        upload_success = minio_connector.upload_binary_data(
            binary_data=image_content, remote_file_path=remote_filename
        )
        # send request info to image info service
        if not upload_success:
            return JSONResponse(
                content={"response": "Error in uploading image to Storage Server."},
                status_code=400,
            )
        else:
            report = {"qoa": "qoa_report"}
            payload = {
                "request_id": request_id,
                "timestamp": rohe_utils.get_current_utc_timestamp(),
                "device_id": device_id,
                "image_url": remote_filename,
                "report": report,
            }

            logger.info(f"This is the payload: {payload}")

            response = requests.post(image_info_service_url, data=payload)
            if response.status_code == 200:
                logger.info(
                    f"\nSuccessfully upload request {request_id} to Image Info Service"
                )
                response = f"Successfully forward the request to the next step. Request id: {request_id}"
                return JSONResponse(content={"response": response}, status_code=200)

            else:
                logger.info(
                    f"\nThis is the response from image info server: {response.json()}"
                )
                response = (
                    f"Failed to upload request {request_id} to Image Info Service"
                )
                return JSONResponse(content={"response": response}, status_code=400)

    except Exception as e:
        logger.exception(f"Error: {e}")
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


# Default GET endpoint
@app.get("/test/")
async def get_test():
    """
    return message to client to notify them that they are accessing the correct ingestion server
    """
    response = "Welcome to Ingestion Server of Object Classification pipeline. Response from GET request"
    return JSONResponse(content={"response": response}, status_code=200)


# Handle the exit signal
def signal_handler(sig, frame):
    logger.info("You pressed Ctrl+C! Gracefully shutting down.")
    consul_client.service_deregister(id=service_id)
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)
