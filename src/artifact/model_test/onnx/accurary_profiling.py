import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
util_directory = os.path.join(current_directory, "../../..", "util")
sys.path.append(util_directory)

dataset_directory = os.path.join(current_directory, "../..", "dataset/imagenet")
sys.path.append(dataset_directory)

from classes import IMAGENET2012_CLASSES  # noqa: E402
from preprocessing import preprocess_input  # noqa: E402

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


def load_model(model_path):
    return ort.InferenceSession(model_path)


def preprocess_image(image_path, input_shape, input_mode):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_shape[1], input_shape[2]))
    # NOTE: careful, the range is different for each models
    image_array = np.array(image).astype("float32")
    image_array = preprocess_input(image_array, mode=input_mode)

    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def get_predictions(model, image_array):
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: image_array})
    return outputs[0]


def evaluate_model(model, image_dir, input_shape, input_mode: str):
    key_list = list(IMAGENET2012_CLASSES.keys())
    incorrect = 0
    for file in tqdm(os.listdir(image_dir)[:1000]):
        if file.endswith(".JPEG"):
            root, _ = os.path.splitext(file)
            _, synset_id = os.path.basename(root).rsplit("_", 1)

            image_path = os.path.join(image_dir, file)
            image_array = preprocess_image(image_path, input_shape, input_mode)
            output = get_predictions(model, image_array)
            predicted_class = np.argmax(output, axis=1)[0]

            key_at_index = key_list[predicted_class]
            if key_at_index != synset_id:
                incorrect += 1

    print(incorrect)
    # # Calculate accuracy
    # accuracy = accuracy_score(true_labels, predictions)
    # return accuracy


model_name = "MobileNet"
model_path = f"./onnx_model/{model_name}.onnx"
image_dir = "../../dataset/imagenet/data/val_images/"
input_shape = MODEL_CONFIG[model_name][0]
input_mode = MODEL_CONFIG[model_name][1]

model = load_model(model_path)

evaluate_model(model, image_dir, input_shape, input_mode)
