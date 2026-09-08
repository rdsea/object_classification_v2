# This's run from root dir
FROM debian:stable-slim

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/

COPY pyproject.toml uv.lock /workspace/

RUN uv sync --frozen --no-install-workspace --package=llm_service --no-cache --compile-bytecode

COPY src/llm_service /workspace/src/llm_service

COPY src/util /workspace/src/util

RUN uv sync --frozen --package=llm_service --no-cache --compile-bytecode

EXPOSE 5012

ENV PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace/src/llm_service

ENTRYPOINT [ "./run_server.sh" ]
