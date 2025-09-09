FROM debian:stable-slim

# Set the working directory in the container
WORKDIR /llm_service

COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /bin/

# Copy the pyproject.toml file to leverage Docker layer caching
COPY ./llm_service/*.toml ./

COPY ./llm_service/run_server.sh ./llm_service/run_server.sh

RUN uv python pin 3.10.18

RUN uv sync --compile-bytecode --no-cache

COPY ./llm_service/*.py ./llm_service/

# Make the run_server.sh script executable
#RUN chmod +x run_server.sh

COPY ./util/ ./util

EXPOSE 5012

WORKDIR /llm_service/llm_service

RUN chmod +x run_server.sh

ENV PATH="/llm_service/.venv/bin:$PATH"

# Command to run the application
ENTRYPOINT [ "./run_server.sh" ]
#CMD ["./run_server.sh"]
