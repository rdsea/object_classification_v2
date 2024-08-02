import asyncio
import logging
import signal
import sys
import time
from typing import Annotated

from aiohttp import ClientSession, FormData
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from rohe.common import rohe_utils
from rohe.service_registry.consul import ConsulClient

config_lock = asyncio.Lock()  # Lock to control access to the global variable

PORT = 5011
config_file = "ensemble_service.yaml"
config = rohe_utils.load_config(file_path=config_file)

assert config is not None
logging.debug(f"Ensemble Service configuration: {config}")

local_ip = rohe_utils.get_local_ip()
consul_client = ConsulClient(
    config["external_services"]["service_registry"]["consul_config"]
)
service_id = consul_client.service_register(
    name="ensemble_service1", address=local_ip, tag=["nii_case"], port=PORT
)


app = FastAPI()
app.state.config = config


def get_service_url(tags, query_type, service_name, consul_client) -> str | None:
    # set flag to check if image info service is available
    service_flag = False
    try:
        # get tags and query type for image info service
        # try 3 times to get image info service
        for _ in range(1, 3):
            service_list: dict = rohe_utils.handle_service_query(
                consul_client=consul_client,
                service_name=service_name,
                query_type=query_type,
                tags=tags,
            )
            if service_list:
                service = service_list[0]
                service_flag = True
                break
            time.sleep(1)
            logging.info(f"No {service_name} service found. Retrying...")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return None
    if not service_flag:
        return None

    # create image info service url
    service_endpoint = "inference"
    service_url = f"http://{service['Address']}:{service['Port']}/{service_endpoint}"
    logging.debug(f"Get image info url successfully: {service_url}")
    return service_url


def get_inference_service_url(ensemble: dict):
    list_service_url = []
    for service_name, metadata in ensemble.items():
        tags = metadata["tags"]
        query_type = metadata["type"]
        service_url = get_service_url(tags, query_type, service_name, consul_client)
        if service_url is not None:
            list_service_url.append(service_url)
    return list_service_url


async def send_post_request(session, url, payload, image_data):
    data = FormData()
    data.add_field(
        "image", image_data, filename="image", content_type="application/octet-stream"
    )
    data.add_field("request_id", payload["request_id"])
    data.add_field("shape", payload["shape"])
    data.add_field("dtype", payload["dtype"])

    async with session.post(url, data=data) as response:
        return await response.json()  # Assuming the response is JSON


async def process_image_task(
    image_data: bytes, request_id: str, shape: str, dtype: str
):
    ensemble = app.state.config["ensemble"]
    # chosen_ensemble_function = getattr(
    #     ensemble_function, app.state.config["aggregating_func"]["func_name"]
    # )
    list_service_url = get_inference_service_url(ensemble)
    logging.info(f"List service url: {list_service_url}")

    if len(list_service_url) != 0:
        payload = {"request_id": request_id, "shape": shape, "dtype": dtype}
        async with ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    send_post_request(session, url, payload, image_data)
                )
                for url in list_service_url
            ]
            # NOTE: asyncio.wait should be use with Task, not coroutine
            done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

            for task in done:
                result = await task
                logging.info(f"Result: {result}")

    else:
        raise RuntimeError("No inference service url")


def background_image_processing(task):
    asyncio.run(task)


@app.post("/ensemble_service/")
async def get_image(request: Request):
    return "OK"
    # try:
    #     logging.info(f"Received request with metadata: {request_id}")
    #     logging.info(f"Received request with shape: {shape}")
    #     logging.info(f"Received request with dtype: {dtype}")
    #     image_bytes = await image.read()
    #     logging.info(f"Received image with type: {type(image)}")
    #
    #     task = process_image_task(image_bytes, request_id, shape, dtype)
    #     background_tasks.add_task(background_image_processing, task)
    #
    #     response = "Success to add image to Ensemble Service"
    #     return JSONResponse(content={"response": response}, status_code=200)
    # except Exception as e:
    #     logging.exception(f"Error: {e}")
    #     return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


@app.post("/change_config/")
async def change_requirement(configuration: Annotated[dict, Form()]):
    try:
        async with config_lock:
            app.state.config = configuration
            response = f"Change ensemble to: {configuration} successfully"
            return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.exception(f"Error: {e}")
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


# Handle the exit signal
def signal_handler(sig, frame):
    logging.info("You pressed Ctrl+C! Gracefully shutting down.")
    consul_client.service_deregister(id=service_id)
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)
