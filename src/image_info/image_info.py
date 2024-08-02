import signal
import sys

import redis
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from rohe.common import rohe_utils
from rohe.common.logger import logger
from rohe.service_registry.consul import ConsulClient

PORT = 5011
try:
    config_file = "image_info_config.yaml"
    config = rohe_utils.load_config(file_path=config_file)
except Exception as e:
    logger.error(f"Error loading config file: {e}")
    sys.exit(1)
assert config is not None
logger.info(f"Image Info configuration: {config}")

# consul for service register
# register service
local_ip = rohe_utils.get_local_ip()
consul_client = ConsulClient(
    config=config["external_services"]["service_registry"]["consul_config"]
)
service_id = consul_client.service_register(
    name="image_info", address=local_ip, tag=["nii_case"], port=PORT
)


# NOTE: initialize dependency before passing to the restful server
redis_config = config["external_services"]["redis_server"]
redis_client = redis.Redis(
    host=redis_config["host"], port=redis_config["port"], db=redis_config["db"]
)

app = FastAPI()


# Create a POST endpoint to add new items
@app.post("/image_info_service/add/")
async def add(
    timestamp: str = Form("timestamp"),
    device_id: str = Form("device_id"),
    request_id: str = Form("request_id"),
    image_url: str = Form("image_url"),
    dtype: str = Form("dtype"),
    shape: str = Form("shape"),
    report: str = Form("report"),
):
    """
    Handles POST requests to coordinate tasks between the ingestion and processing instances.

    Commands:
    1. add: Called by the ingestion instance to add an image to the queue for processing.
        Adds the image to the unprocessed_images list in Redis.

    2. complete: Called by the processing instance to indicate that an image has been fully processed.
        Removes the image metadata from the processing_images list and adds it to the processed_images list in Redis.

    :return: JSON response indicating the status of the command or an error message.
    """
    try:
        logger.info(
            f"Received request with request_id: {request_id}, timestamp: {timestamp}, device_id: {device_id}, image_url: {image_url}, dtype: {dtype}, shape: {shape}, report: {report}"
        )

        image_info = {
            "request_id": request_id,
            "timestamp": timestamp,
            "device_id": device_id,
            "image_url": image_url,
            "dtype": dtype,
            "shape": shape,
        }
        required_fields = ["request_id", "timestamp", "device_id", "image_url"]
        if all(image_info[field] is not None for field in required_fields):
            serialized_image_info = rohe_utils.message_serialize(image_info)
            redis_client.lpush("unprocessed_images", serialized_image_info)
            response = f"Success to add image {request_id} to Image Info Service"
            return JSONResponse(content={"response": response}, status_code=200)
        else:
            logger.info(
                f"This is the image info that does not satisfy the requirement of having all the field: {image_info}"
            )
            response = f"Some required fields are missing, the required field are: {required_fields}"
            return JSONResponse(content={"response": response}, status_code=400)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


@app.post("/image_info_service/complete/")
async def complete(
    timestamp: str = Form("timestamp"),
    device_id: str = Form("device_id"),
    request_id: str = Form("request_id"),
    image_url: str = Form("image_url"),
    dtype: str = Form("dtype"),
    shape: str = Form("shape"),
    report: str = Form("report"),
):
    """
    Handles POST requests to coordinate tasks between the ingestion and processing instances.

    Commands:
    1. add: Called by the ingestion instance to add an image to the queue for processing.
        Adds the image to the unprocessed_images list in Redis.

    2. complete: Called by the processing instance to indicate that an image has been fully processed.
        Removes the image metadata from the processing_images list and adds it to the processed_images list in Redis.

    :return: JSON response indicating the status of the command or an error message.
    """
    try:
        logger.info(
            f"Received request with request_id: {request_id}, timestamp: {timestamp}, device_id: {device_id}, image_url: {image_url}, dtype: {dtype}, shape: {shape}, report: {report}"
        )

        image_info = {
            "request_id": request_id,
            "timestamp": timestamp,
            "device_id": device_id,
            "image_url": image_url,
            "dtype": dtype,
            "shape": shape,
        }
        serialized_image_info = rohe_utils.message_serialize(image_info)
        result = redis_client.lrem("processing_images", 0, serialized_image_info)
        logger.info(f"This is the result: {result}")
        if result >= 1:
            redis_client.lpush("processed_images", serialized_image_info)
            response = f"Success to processed image {request_id} in Image Info Service"
            return JSONResponse(content={"response": response}, status_code=200)
        response = f"Fail to processed image {request_id} in Image Info Service"
        return JSONResponse(content={"response": response}, status_code=400)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


# Default GET endpoint
@app.get("/image_info_service/task/")
async def get_test():
    """
    Handles GET requests from the processing instances to claim an image for processing.

    Functionality:
    - Moves an image metadata from the unprocessed_images list to the processing_images list in Redis.
    - Claims the image for a specific processing instance.

    :return: JSON response containing the claimed image or a status indicating no unprocessed images.
    """
    serialized_image_info = redis_client.rpoplpush(
        "unprocessed_images", "processing_images"
    )
    if serialized_image_info:
        image_info = rohe_utils.message_deserialize(serialized_image_info)
        return JSONResponse(content={"image_info": image_info}, status_code=200)

    else:
        response = "no unprocessed images"
        return JSONResponse(content={"response": response}, status_code=202)


# NOTE: Handle the exit signal
def signal_handler(sig, frame):
    logger.info("You pressed Ctrl+C! Gracefully shutting down.")
    consul_client.service_deregister(id=service_id)
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)
