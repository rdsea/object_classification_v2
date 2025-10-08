# This's run from root dir
FROM debian:stable-slim

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/

COPY pyproject.toml uv.lock /workspace/

RUN uv sync --frozen --no-install-workspace --package=ensemble --no-cache --compile-bytecode

COPY src/ensemble /workspace/src/ensemble

COPY src/util /workspace/src/util

RUN uv sync --frozen --package=ensemble --no-cache --compile-bytecode

EXPOSE 5011

ENV PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace/src/ensemble

ENTRYPOINT [ "./run_server.sh" ]

