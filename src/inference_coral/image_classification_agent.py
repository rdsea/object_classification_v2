import os
import sys

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from datamodel import (
    ImageClassificationModelEnum,
    ModelConfig,
)
from numpy._typing import NDArray

current_directory = os.path.dirname(os.path.abspath(__file__))
util_directory = os.path.join(current_directory, "..", "util")
sys.path.append(util_directory)

from classes import IMAGENET2012_CLASSES  # noqa: E402
from preprocessing import preprocess_input  # noqa: E402

key_list = list(IMAGENET2012_CLASSES.keys())


class ImageClassificationAgent:
    def __init__(
        self,
        chosen_model: ImageClassificationModelEnum,
        model_config: ModelConfig,
    ):
        self.interpreter = tflite.Interpreter(
            model_path=f"./tflite_cpu_model/{chosen_model.name}.tflite"
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.model_config = model_config

    def reshape(self, image_array: NDArray, enlarge: bool):
        # NOTE: interpolation choice taken from https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
        reshaped_image = cv2.resize(
            image_array,
            self.model_config.input_shape[1:2],
            interpolation=cv2.INTER_LINEAR if enlarge else cv2.INTER_AREA,
        )
        return reshaped_image

    def predict(self, image_array: NDArray):
        if image_array.shape[1] != self.model_config.input_shape[1]:
            image_array = self.reshape(
                image_array, image_array.shape[1] > self.model_config.input_shape[1]
            )

        image_array = preprocess_input(image_array, mode=self.model_config.input_mode)

        if len(image_array.shape) != 4:
            image_array = np.expand_dims(image_array, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], image_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        outputs = np.squeeze(output_data)

        predicted_class_index = np.argmax(outputs, axis=1)[0]
        return key_list[predicted_class_index], float(outputs[0][predicted_class_index])
