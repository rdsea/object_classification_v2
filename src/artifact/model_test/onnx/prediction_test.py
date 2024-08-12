import time

import numpy as np
import onnxruntime as ort


def load_model(onnx_model_path):
    # Set up the session options and enable the CUDA Execution Provider
    session_options = ort.SessionOptions()

    # Initialize the InferenceSession with the ONNX model and specify the CUDA Execution Provider
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["CUDAExecutionProvider"],
        sess_options=session_options,
    )

    return session


def prepare_input_data():
    # Prepare your input data
    # This is a placeholder for actual input data that matches your model's input requirements
    # Replace with appropriate data loading and preprocessing
    input_data = np.random.rand(1, 224, 224, 3).astype(
        np.float32
    )  # Example input for an image classification model

    # Create input dictionary: {'input_name': input_data}
    input_name = "serving_default_input_11:0"  # Replace with your actual input name
    inputs = {input_name: input_data}
    return inputs


def run_inference(session, inputs):
    # Run inference
    start_time = time.time()
    outputs = session.run(None, inputs)
    print(f"Latency {(time.time()-start_time)*1000} ms")
    return outputs


def main():
    # Load the ONNX model
    onnx_model_path = "VGGG16.onnx"  # Replace with your model's path
    session = load_model(onnx_model_path)

    # Prepare input data
    inputs = prepare_input_data()

    # Run inference
    for _ in range(10):
        _ = run_inference(session, inputs)


if __name__ == "__main__":
    main()
