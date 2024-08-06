FROM python:3.10.14-bullseye

WORKDIR /tflite_latency_test

RUN python3 -m pip install tflite-runtime
RUN pip install numpy==1.23.5

COPY ./tflite_model/MobileNet.tflite ./tflite_model/MobileNet.tflite
COPY ./tflite_latency_test.py ./tflite_latency_test.py

CMD ["python3", "tflite_latency_test.py"]
