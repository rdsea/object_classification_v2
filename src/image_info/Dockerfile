FROM python:3.10

WORKDIR /image_info
COPY ./requirements.txt .

RUN apt-get update && \
  apt-get install -y wget vim gcc python3-dev &&\
  pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5011
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5011", "image_info:app"]

