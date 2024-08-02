FROM python:3.10

WORKDIR /processing
COPY ./requirements.txt .

RUN apt-get update && \
  apt-get install -y wget vim gcc python3-dev &&\
  pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5010
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5010", "processing:app"]

