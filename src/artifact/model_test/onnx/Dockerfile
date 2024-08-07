FROM python:3.10.14-slim-bullseye

WORKDIR /tflite_latency_test

RUN pip install onnxruntime

COPY ./tflite_model/MobileNet.tflite ./tflite_model/MobileNet.tflite
COPY ./tflite_latency_test.py ./tflite_latency_test.py

CMD ["python3", "tflite_latency_test.py"]
