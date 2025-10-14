import asyncio
import logging
import os
import sys
from uuid import uuid4

import aiohttp
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from image_processing_functions import resize

from util.utils import load_config, setup_otel

SERVICE_NAME = os.environ.get("SERVICE_NAME", "ensemble")

ENSEMBLE_SERVICE_URL = (
    "http://ensemble-service.default.svc.cluster.local:5011/ensemble_service"
)
if os.environ.get("DOCKER"):
    ENSEMBLE_SERVICE_URL = "http://ensemble:5011/ensemble_service"

if os.environ.get("OPENZITI"):
    ENSEMBLE_SERVICE_URL = "http://ensemble.miniziti.private:5011/ensemble_service"


setup_otel(SERVICE_NAME)

try:
    config_file = "preprocessing_config.yaml"
    config = load_config(file_path=config_file)
except Exception as e:
    logging.error(f"Error loading config file: {e}")
    sys.exit(1)
assert config is not None


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


app = FastAPI()


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
        logging.error(content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"UploadFile with type {content_type} is not accepted, only accept {accepted_file_types}",
        )


@app.post("/preprocessing")
async def processing_image(file: UploadFile, request: Request):
    logging.debug(request.headers)
    validate_image_type(file.content_type)

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    # NOTE: cv2 and Pillow has different color channel layout
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = image.shape
    if shape != (224, 224, 3):
        processed_image = resize(image)
    else:
        processed_image = image

    image_bytes = processed_image.tobytes()
    request_id = str(uuid4())

    headers = {
        "Timestamp": request.headers.get("Timestamp"),
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(image_bytes)),
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            logging.debug(ENSEMBLE_SERVICE_URL)
            async with session.post(
                headers=headers,
                url=ENSEMBLE_SERVICE_URL,
                data=image_bytes,
                params={"request_id": request_id},
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to send image to ensemble service. Status code: {response.status}",
                    )
                _ = await response.json()
        return "File accepted"
    except aiohttp.ClientError as e:
        logging.error(f"Client error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to connect to ensemble service.",
        )
    except asyncio.TimeoutError:
        logging.error("Request to ensemble service timed out.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Request to ensemble service timed out.",
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


if os.environ.get("MANUAL_TRACING"):
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    AioHttpClientInstrumentor().instrument()
    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
