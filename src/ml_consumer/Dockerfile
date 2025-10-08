FROM debian:stable-slim

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/

COPY pyproject.toml uv.lock /workspace/

RUN uv sync --frozen --no-install-workspace --package=ml-consumer --no-cache --compile-bytecode

COPY src/ml_consumer /workspace/src/ml_consumer

COPY src/util /workspace/src/util

RUN uv sync --frozen --package=ml-consumer --no-cache --compile-bytecode

ENV PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace/src/ml_consumer

CMD ["python", "ml_consumer.py"]
