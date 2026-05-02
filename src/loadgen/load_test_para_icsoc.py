from locust import HttpUser, between, task, events
from gevent.lock import Semaphore

import csv
import logging
import os
import random
import time
import uuid
from pathlib import Path


csv_file = None
csv_writer = None
csv_lock = Semaphore()


class ImageUploadUser(HttpUser):
    wait_time = between(1, 1)

    def on_start(self):
        self.ds_path = Path(self.environment.parsed_options.ds_path)
        self.device_id = self.environment.parsed_options.device_id
        self.run_id = self.environment.parsed_options.run_id

        try:
            files = os.listdir(self.ds_path)
            self.jpeg_images_list = [
                file for file in files if file.lower().endswith(".jpeg")
            ]
        except Exception as e:
            logging.error(f"Failed to list dataset directory: {e}")
            self.jpeg_images_list = []

    @task
    def upload_image(self):
        global csv_writer

        if not self.jpeg_images_list:
            logging.warning("No JPEG images found, skipping task.")
            return

        request_id = str(uuid.uuid4())
        sensor_send_time = time.time()
        sensor_receive_time = None
        response_time_ms = None
        status_code = None
        success = False
        error = ""
        prediction = ""

        try:
            random_image = random.choice(self.jpeg_images_list)
            image_path = os.path.join(self.ds_path, random_image)

            root, _ = os.path.splitext(image_path)
            _, synset_id = os.path.basename(root).rsplit("_", 1)

            with open(image_path, "rb") as img_file:
                img_data = img_file.read()

            files = {
                "file": ("random_image.jpeg", img_data, "image/jpeg"),
                "device_id": (None, self.device_id),
                "request_id": (None, request_id),
                "run_id": (None, self.run_id),
            }

            headers = {
                "Timestamp": str(sensor_send_time),
                "X-Request-ID": request_id,
                "X-Run-ID": self.run_id,
                "X-Device-ID": self.device_id,
            }

            with self.client.post(
                "/preprocessing",
                files=files,
                catch_response=True,
                headers=headers,
            ) as response:
                sensor_receive_time = time.time()
                response_time_ms = (sensor_receive_time - sensor_send_time) * 1000
                status_code = response.status_code

                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        prediction = str(json_response)
                    except Exception:
                        prediction = response.text[:300]

                    success = True
                    response.success()
                else:
                    error = f"Failed with {response.status_code}: {response.text[:300]}"
                    response.failure(error)

        except Exception as e:
            sensor_receive_time = time.time()
            response_time_ms = (sensor_receive_time - sensor_send_time) * 1000
            error = str(e)
            logging.exception(f"Exception during upload: {e}")

        finally:
            if csv_writer is not None:
                with csv_lock:
                    csv_writer.writerow(
                        {
                            "run_id": self.run_id,
                            "request_id": request_id,
                            "device_id": self.device_id,
                            "image_name": random_image
                            if "random_image" in locals()
                            else "",
                            "synset_id": synset_id if "synset_id" in locals() else "",
                            "sensor_send_time": sensor_send_time,
                            "sensor_receive_time": sensor_receive_time,
                            "locust_response_time_ms": response_time_ms,
                            "status_code": status_code,
                            "success": success,
                            "prediction": prediction,
                            "error": error,
                        }
                    )
                    csv_file.flush()


def add_custom_arguments(parser):
    parser.add_argument(
        "--ds-path",
        type=str,
        env_var="LOCUST_DS_PATH",
        default="./image/",
        help="Path to dataset folder containing JPEG images",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        env_var="LOCUST_DEVICE_ID",
        default="drone_1",
        help="Device ID to send with each request",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        env_var="LOCUST_RUN_ID",
        default=f"manual_{int(time.time())}",
        help="Unique run ID for joining Locust and MongoDB results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        env_var="LOCUST_RESULTS_DIR",
        default="~/results",
        help="Directory for detailed Locust per-request CSV",
    )


@events.init_command_line_parser.add_listener
def _(parser):
    add_custom_arguments(parser)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global csv_file, csv_writer

    run_id = environment.parsed_options.run_id
    results_dir = Path(environment.parsed_options.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    detail_csv_path = results_dir / f"{run_id}_locust_requests.csv"

    csv_file = open(detail_csv_path, "a", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "run_id",
            "request_id",
            "device_id",
            "image_name",
            "synset_id",
            "sensor_send_time",
            "sensor_receive_time",
            "locust_response_time_ms",
            "status_code",
            "success",
            "prediction",
            "error",
        ],
    )

    if csv_file.tell() == 0:
        csv_writer.writeheader()

    logging.info(f"Writing per-request Locust results to {detail_csv_path}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    global csv_file

    if csv_file is not None:
        csv_file.flush()
        csv_file.close()
        csv_file = None
