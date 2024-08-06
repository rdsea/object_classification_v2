import os
import time

import numpy as np

# import tensorflow as tf
import tflite_runtime.interpreter as tflite

# from tensorflow.keras.applications.resnet50 import (
#     ResNet50,
#     decode_predictions,
#     preprocess_input,
# )


path = "./tflite_model/"
dir_list = os.listdir(path)
for dir in dir_list:
    saved_model_path = path + dir
    interpreter = tflite.Interpreter(model_path=saved_model_path)

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    avg_time = 0
    for _ in range(10):
        start_time = time.time()
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        avg_time += time.time() - start_time

    print(f"{dir} runtime: {avg_time / 10}")
