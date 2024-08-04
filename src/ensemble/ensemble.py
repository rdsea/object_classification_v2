import asyncio
import logging
import os
import signal
import sys
import time
from typing import Annotated

import aiohttp
import ensemble_function
from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import JSONResponse
from rohe.common import rohe_utils
from rohe.service_registry.consul import ConsulClient

config_lock = asyncio.Lock()  # Lock to control access to the global variable


PORT = int(os.environ["PORT"])

config_file = "ensemble_service.yaml"
config = rohe_utils.load_config(file_path=config_file)

assert config is not None
logging.debug(f"Ensemble Service configuration: {config}")

local_ip = rohe_utils.get_local_ip()
consul_client = ConsulClient(
    config["external_services"]["service_registry"]["consul_config"]
)
service_id = consul_client.service_register(
    name="ensemble", address=local_ip, tag=["nii_case"], port=PORT
)


app = FastAPI()
app.state.config = config


def get_service_url(tags, query_type, service_name, consul_client) -> str | None:
    try:
        for _ in range(1, 3):
            service_list: dict = rohe_utils.handle_service_query(
                consul_client=consul_client,
                service_name=service_name,
                query_type=query_type,
                tags=tags,
            )
            if service_list:
                inference_service = service_list[0]
                return f"http://{inference_service['Address']}:{inference_service['Port']}/inference/"
            time.sleep(1)
            logging.info(f"No {service_name} service found. Retrying...")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return None
    return None


def get_inference_service_url(ensemble: dict):
    list_service_url = []
    for service_name, metadata in ensemble.items():
        tags = metadata["tags"]
        query_type = metadata["type"]
        service_url = get_service_url(tags, query_type, service_name, consul_client)
        if service_url is not None:
            list_service_url.append(service_url)
    return list_service_url


async def send_post_request(
    session: aiohttp.ClientSession, url: str, image_data: bytes
):
    async with session.post(url, data=image_data) as response:
        return await response.json()  # Assuming the response is JSON


async def process_image_task(image_data: bytes, request_id: str):
    ensemble = app.state.config["ensemble"]
    chosen_ensemble_function = getattr(
        ensemble_function,
        app.state.config["aggregating"]["aggregating_func"]["func_name"],
    )
    list_service_url = get_inference_service_url(ensemble)
    logging.info(f"List service url: {list_service_url}")

    if list_service_url:
        async with aiohttp.ClientSession(trust_env=True) as session:
            tasks = [
                asyncio.create_task(send_post_request(session, url, image_data))
                for url in list_service_url
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

            results = []
            for task in done:
                results.append(await task)
            print(chosen_ensemble_function(results, request_id))

    else:
        raise RuntimeError("No inference service url")


@app.post("/ensemble_service/")
async def ensemble(
    request: Request,
    background_tasks: BackgroundTasks,
):
    try:
        image_bytes = await request.body()
        request_id = request.query_params["request_id"]
        # logging.info(image_bytes)
        background_tasks.add_task(process_image_task, image_bytes, request_id)

        response = "Success to add image to Ensemble Service"
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.exception(f"Error: {e}")
        return JSONResponse(content={"error": f"Error: {e}"}, status_code=500)


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
