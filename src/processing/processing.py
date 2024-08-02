import asyncio
import os
import signal
import sys
import time

import aiohttp
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from image_processing_functions import (
    resize_and_pad,
)
from rohe.common import rohe_utils
from rohe.common.logger import logger
from rohe.service_registry.consul import ConsulClient
from rohe.storage.minio import MinioConnector

config_lock = asyncio.Lock()  # Lock to control access to the global variable
PORT = 5010
try:
    port = PORT
    config_file = "processing_config.yaml"
    config = rohe_utils.load_config(file_path=config_file)
except Exception as e:
    logger.error(f"Error loading config file: {e}")
    sys.exit(1)
assert config is not None
logger.info(f"Image Processing configuration: {config}")

minio_connector = MinioConnector(config["external_services"]["minio_storage"])

local_ip = rohe_utils.get_local_ip()
consul_client = ConsulClient(
    config=config["external_services"]["service_registry"]["consul_config"]
)
service_id = consul_client.service_register(
    name="processing", address=local_ip, tag=["nii_case"], port=port
)


accepted_file_types = [
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
]


def get_ensemble_service_url() -> str | None:
    try:
        # get tags and query type for image info service
        tags = config["external_services"]["inference_service"]["tags"]
        query_type = config["external_services"]["inference_service"]["type"]
        ensemble_name = config["external_services"]["inference_service"][
            "ensemble_name"
        ]

        # try 3 times to get image info service
        for _ in range(1, 3):
            ensemble_service_list: dict = rohe_utils.handle_service_query(
                consul_client=consul_client,
                service_name=ensemble_name,
                query_type=query_type,
                tags=tags,
            )
            if ensemble_service_list:
                ensemble_service = ensemble_service_list[0]
                ensemble_endpoint = "ensemble_service"
                ensemble_service_url = f"http://{ensemble_service['Address']}:{ensemble_service['Port']}/{ensemble_endpoint}"
                logger.debug(f"Get ensemble url successfully: {ensemble_service_url}")
                return ensemble_service_url
            time.sleep(1)
            logger.info("Waiting for image info service to be available")
        return None
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return None


app = FastAPI()


# @app.post("/change_ensemble/")
# async def change_ensemble(ensemble_name: str = Form("ensemble_name")):
#     try:
#         async with config_lock:
#             executor.config["external_services"]["inference_service"][
#                 "ensemble_name"
#             ] = ensemble_name
#             response = f"Change ensemble to: {ensemble_name} successfully"
#             return JSONResponse(content={"response": response}, status_code=200)
#     except Exception as e:
#         logger.error(f"Error: {e}", exc_info=True)
#         return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)
#
#
# @app.post("/change_config/")
# async def change_requirement(configuration: Annotated[dict, Form()]):
#     try:
#         async with config_lock:
#             executor.config = configuration
#             response = f"Change ensemble to: {configuration} successfully"
#             return JSONResponse(content={"response": response}, status_code=200)
#     except Exception as e:
#         logger.error(f"Error: {e}", exc_info=True)
#         return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)
#
#
@app.get("/test/")
async def get_test():
    """
    return message to client to notify them that they are accessing the correct ingestion server
    """
    response = "This is processing service controller"
    return JSONResponse(content={"response": response}, status_code=200)


def validate_image_type(content_type: str | None):
    if content_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="UploadFile have no content type",
        )
    if content_type not in accepted_file_types:
        logger.info(content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"UploadFile with type {content_type} is not accepted, only accept {accepted_file_types}",
        )


@app.post("/processing/")
async def processing_image(file: UploadFile):
    validate_image_type(file.content_type)

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    # NOTE: I don't know why we do this
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = image.shape

    logger.info(shape)
    if shape != (32, 32, 3):
        processed_image: np.ndarray = resize_and_pad(image)
    else:
        processed_image = image

    save_directory = "processed_images"
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, file.filename)
    cv2.imwrite(save_path, processed_image)
    # logger.info(f"Image saved to {save_path}")
    ensemble_service_url = get_ensemble_service_url()
    if ensemble_service_url is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Can't find ensemble service url",
        )
    image_bytes = processed_image.tobytes()

    async with aiohttp.ClientSession() as session:
        logger.info(ensemble_service_url)
        async with session.post(ensemble_service_url, data=image_bytes) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to send image to ensemble service. Status code: {response.status}",
                )
            _ = await response.json()
    return "File accepted"


def signal_handler(sig, frame):
    logger.info("You pressed Ctrl+C! Gracefully shutting down.")
    consul_client.service_deregister(id=service_id)
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)
