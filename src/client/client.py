import argparse
import os
import random
import time

import requests


def send_request(config):
    config["rate"]
    data_path = config["test_ds"]

    files = os.listdir(data_path)
    png_images = [file for file in files if file.lower().endswith(".png")]
    for _ in range(1, 10000000000000):
        # data = {
        #     "timestamp": str(time.time()),
        #     "device_id": config["device_id"],
        # }
        # with open("path/to/your/image", "rb") as img:
        #     files = {"image": img}
        #     response = requests.post(url, files=files)
        #     print(response.json())
        #
        random_image = random.choice(png_images)
        image_path = os.path.join(data_path, random_image)

        with open(image_path, "rb") as img_file:
            files = {"file": (random_image, img_file, "image/png")}
            response = requests.post(config["url"], files=files)

        print(response.status_code)
        print(response.json())
        time.sleep(10)


if __name__ == "__main__":
    # init_env_variables()
    parser = argparse.ArgumentParser(
        description="Argument for choosingg model to request"
    )
    # parser.add_argument('--test_ds', type= str, help='default test dataset path',
    #             default= "01.jpg")
    parser.add_argument(
        "--test_ds",
        type=str,
        help="test dataset path",
        default="./images/",
    )
    parser.add_argument(
        "--rate", type=int, help="number of requests per second", default=20
    )
    parser.add_argument(
        "--device_id", type=str, help="specify device id", default="aaltosea_cam_01"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="request url",
        default="http://localhost:5010/processing",
    )

    # Parse the parameters
    args = parser.parse_args()
    device_id = args.device_id
    test_ds = args.test_ds
    req_rate = args.rate

    config = {
        "device_id": device_id,
        "test_ds": test_ds,
        "rate": args.rate,
        "url": args.url,
    }
    send_request(config)
