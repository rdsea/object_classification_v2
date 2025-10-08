# This's run from root dir
FROM rdsea/debian-slim-opencv:latest

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/

COPY pyproject.toml uv.lock /workspace/

RUN uv sync --frozen --no-install-workspace --package=preprocessing --no-cache --compile-bytecode

COPY src/preprocessing /workspace/src/preprocessing

COPY src/util /workspace/src/util

RUN uv sync --frozen --package=preprocessing --no-cache --compile-bytecode

EXPOSE 5010

ENV PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace/src/preprocessing

ENTRYPOINT [ "./run_server.sh" ]

