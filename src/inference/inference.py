import logging
import os
import sys

import numpy as np
import yaml
from datamodel import ImageClassificationModelEnum, InferenceServiceConfig
from fastapi import FastAPI, Request
from image_classification_agent import ImageClassificationAgent

from util.utils import setup_otel

current_directory = os.path.dirname(os.path.abspath(__file__))
util_directory = os.path.join(current_directory, "..", "util")
sys.path.append(util_directory)

# Set up logging with service name and instance
default_model = "MobileNetV2"
SERVICE_NAME = os.environ.get(
    "SERVICE_NAME", f"inference-{os.environ.get('CHOSEN_MODEL', default_model).lower()}"
)
setup_otel(SERVICE_NAME)

chosen_model = os.environ["CHOSEN_MODEL"]
# NOTE: model config in the inference service config
#
MODEL_CONFIG = {
    "DenseNet121": [(1, 224, 224, 3), "torch"],
    "DenseNet201": [(1, 224, 224, 3), "torch"],
    "EfficientNetB0": [(1, 224, 224, 3), "raw"],
    "EfficientNetB7": [(1, 600, 600, 3), "raw"],
    "EfficientNetV2L": [(1, 480, 480, 3), "raw"],
    "EfficientNetV2S": [(1, 384, 384, 3), "raw"],
    "InceptionResNetV2": [(1, 299, 299, 3), "tf"],
    "InceptionV3": [(1, 299, 299, 3), "tf"],
    "MobileNet": [(1, 224, 224, 3), "tf"],
    "MobileNetV2": [(1, 224, 224, 3), "tf"],
    "NASNetLarge": [(1, 331, 331, 3), "tf"],
    "NASNetMobile": [(1, 224, 224, 3), "tf"],
    "ResNet50": [(1, 224, 224, 3), "caffe"],
    "ResNet50V2": [(1, 224, 224, 3), "tf"],
    "VGG16": [(1, 224, 224, 3), "caffe"],
    "Xception": [(1, 299, 299, 3), "tf"],
}

try:
    with open("./inference_service_config.yaml") as file:
        yaml_content = yaml.safe_load(file)  # Load YAML content
        config = InferenceServiceConfig(**yaml_content)  # Parse using Pydantic model
except Exception as e:
    logging.error(f"Error loading config file: {e}")
    sys.exit(1)
logging.info(f"Inference configuration: {config}")

chosen_model = ImageClassificationModelEnum[chosen_model]

model_config = config.model_config_dict[chosen_model]
ml_agent = ImageClassificationAgent(chosen_model, model_config)

app = FastAPI()


@app.post("/inference")
async def inference(request: Request):
    image_bytes = await request.body()

    # headers = dict(request.headers)
    # carrier = {"traceparent": headers["traceparent"]}
    # ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    # print(f"Received context: {ctx}")
    #
    # b2 = {"baggage": headers["baggage"]}
    # ctx2 = W3CBaggagePropagator().extract(b2, context=ctx)
    # print(f"Received context2: {ctx2}")
    # logging.info(image_bytes)
    # with tracer.start_span("inference"):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # # NOTE: Here we assume that the processing service has reshape the input image to size 224,224,3
    reconstructed_image = image_array.reshape((224, 224, 3))
    return ml_agent.predict(reconstructed_image)


if os.environ.get("MANUAL_TRACING"):
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
