import os
import time
from collections.abc import Callable

import cv2
import image_processing_functions
import numpy as np
import requests
from rohe.common import rohe_utils
from rohe.common.logger import logger
from rohe.service_registry.consul import ConsulClient
from rohe.storage.minio import MinioConnector


class ProcessingServiceExecutor:
    def __init__(
        self, config: dict, minio_connector: MinioConnector, consul_client: ConsulClient
    ):
        self.config = config

        self.minio_connector = minio_connector
        self.consul_client = consul_client

        self.min_waiting_period: int = config["processing"]["request"]["retry_delay"][
            "min"
        ]
        self.max_waiting_period: int = config["processing"]["request"]["retry_delay"][
            "max"
        ]
        self.request_rate: int = config["processing"]["request"]["rate_per_second"]

        self.image_processing_func: Callable = getattr(
            image_processing_functions,
            config["processing"]["image_processing"]["func_name"],
        )

        self.image_dim = config["processing"]["image_processing"]["target_dim"]

        # create a temp folder to store temporary image file download from minio server
        self.tmp_folder = "tmp_image_folder"
        if not os.path.exists(self.tmp_folder):
            os.mkdir(self.tmp_folder)

    def get_ensemble_service_url(self) -> str | None:
        try:
            # get tags and query type for image info service
            tags = self.config["external_services"]["inference_service"]["tags"]
            query_type = self.config["external_services"]["inference_service"]["type"]
            ensemble_name = self.config["external_services"]["inference_service"][
                "ensemble_name"
            ]

            # try 3 times to get image info service
            for _ in range(1, 3):
                ensemble_service_list: dict = rohe_utils.handle_service_query(
                    consul_client=self.consul_client,
                    service_name=ensemble_name,
                    query_type=query_type,
                    tags=tags,
                )
                if ensemble_service_list:
                    ensemble_service = ensemble_service_list[0]
                    self.ensemble_flag = True
                    ensemble_endpoint = "ensemble_service"
                    ensemble_service_url = f"http://{ensemble_service['Address']}:{ensemble_service['Port']}/{ensemble_endpoint}"
                    logger.debug(
                        f"Get ensemble url successfully: {ensemble_service_url}"
                    )
                    return ensemble_service_url
                time.sleep(1)
                logger.info("Waiting for image info service to be available")
            return None

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return None

    def _processing(self, task: dict, image_info_service_url: str):
        try:
            temp_local_path = self._download_image_from_minio(
                image_url=task["image_url"]
            )
            # if download fail, return None
            if not temp_local_path:
                logger.warning("Fail to download the image")
                return None
        except Exception as e:
            logger.error(f"Error in downloading image from minio: {e}", exc_info=True)
            return None

        task["temp_local_path"] = temp_local_path
        logger.info("Save image to temp folder successfully")

        # process the image (in npy format or other type)
        processed_image = self._image_processing(task)

        del task["temp_local_path"]

        # if fail to process the image
        if processed_image is None:
            logger.error("Fail to process the image")
            return None

        # Notify image info service of completion
        form_payload = task
        response = requests.post(
            image_info_service_url + "/complete/", data=form_payload
        )
        if response.status_code != 200:
            logger.error(
                f"Failed to notify Image Info Service the completion of task {task}"
            )
            return None
        else:
            # Make a POST request to the inference servers
            logger.info(f"Notify Image Info Service the completion of task {task}")

            # To do: make request to inference server
            # Convert the numpy array to bytes
            image_bytes = processed_image.tobytes()
            # Metadata and command
            shape_str = ",".join(map(str, processed_image.shape))
            metadata = {
                "request_id": task["request_id"],
                "shape": shape_str,
                "dtype": str(processed_image.dtype),
            }
            logger.info(f"This is the metadata: {metadata}")
            files = {"image": ("image", image_bytes, "application/octet-stream")}
            try:
                ensemble_service_url = self.get_ensemble_service_url()
                if ensemble_service_url is None:
                    logger.error("No ensemble service url")
                    raise RuntimeError("No ensemble service url")

                logger.info(f"Make request to ensemble servers: {ensemble_service_url}")
                response = requests.post(
                    ensemble_service_url, data=metadata, files=files
                )
                logger.info(
                    f"This is the response from ensemble service: {response.text}"
                )
            except Exception as e:
                logger.exception(f"Error when making request to ensemble service: {e}")

            return True

    def _image_processing(self, task: dict) -> np.ndarray | None:
        try:
            # get image extension
            image_extension = rohe_utils.extract_file_extension(task["image_url"])
            logger.info(f"Get image extension: {image_extension}")

            if image_extension == "npy":
                with open(task["temp_local_path"], "rb") as file:
                    raw_data = file.read()
                # reshape the image and type cast it to the original type
                shape = rohe_utils.convert_str_to_tuple(task["shape"])
                image = np.frombuffer(raw_data, dtype=task["dtype"]).reshape(shape)
            else:
                # read image in png or jpg format
                image = cv2.imread(task["temp_local_path"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                shape = image.shape

            if shape != self.image_dim:
                # resize the image
                processed_image: np.ndarray = self.image_processing_func(
                    image, self.image_dim
                )
            else:
                processed_image = image

            logger.info("Image processing successfully done")
            return processed_image
        except Exception as e:
            logger.error(f"This is the error in image process stage: {e}")
            return None
