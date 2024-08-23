FROM python:3.10.14-slim-bullseye

WORKDIR /processing
COPY ./requirements.txt .

RUN apt-get update && \
  apt-get install -y wget vim gcc python3-dev &&\
  pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5010

ENTRYPOINT [ "./run_server.sh" ]

