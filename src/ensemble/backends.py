"""Resolution of the inference backends the ensemble fans out to.

The set of backends comes from one of two places, in priority order:

1. The ``INFERENCE_BACKENDS`` environment variable, which states every backend
   explicitly as ``name=kind=url``. This is the docker-compose / ConfigMap knob:
   changing the ensemble members no longer requires rebuilding the image.
2. Otherwise the ``ensemble`` list in ``ensemble_service.yaml``, whose entries are
   plain model names turned into URLs by the deployment mode (k8s / DOCKER /
   OPENZITI). This is the legacy path and is kept so existing manifests that only
   mount the YAML keep working.

``kind`` is carried explicitly rather than sniffed from the hostname, because it
decides how the image payload is prepared: ``cnn`` backends receive the raw RGB
buffer produced by preprocessing, ``llm`` backends receive JPEG-encoded bytes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

KIND_CNN = "cnn"
KIND_LLM = "llm"
VALID_KINDS = (KIND_CNN, KIND_LLM)

ENV_VAR = "INFERENCE_BACKENDS"

DEFAULT_PORT = 5012
DEFAULT_PATH = "/inference"


@dataclass(frozen=True)
class Backend:
    """One inference service the ensemble sends every image to."""

    name: str
    kind: str
    url: str

    @property
    def is_llm(self) -> bool:
        return self.kind == KIND_LLM


def _split_entries(raw: str) -> list[str]:
    """Split on commas and newlines so YAML block scalars stay readable."""
    return [
        entry.strip() for entry in raw.replace("\n", ",").split(",") if entry.strip()
    ]


def parse_backend_spec(raw: str) -> list[Backend]:
    """Parse ``name=kind=url`` entries out of the INFERENCE_BACKENDS value.

    ``url`` may itself contain ``=`` (query parameters), so only the first two
    separators are significant.
    """
    backends = []
    for entry in _split_entries(raw):
        parts = entry.split("=", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid {ENV_VAR} entry {entry!r}: expected 'name=kind=url' "
                f"where kind is one of {VALID_KINDS}"
            )
        name, kind, url = (part.strip() for part in parts)
        kind = kind.lower()
        if kind not in VALID_KINDS:
            raise ValueError(
                f"Invalid kind {kind!r} for backend {name!r}: expected one of {VALID_KINDS}"
            )
        if not name or not url:
            raise ValueError(f"Invalid {ENV_VAR} entry {entry!r}: empty name or url")
        backends.append(Backend(name=name, kind=kind, url=url))
    return backends


def _legacy_host(name: str) -> str:
    """Hostname a bare model name maps to, per deployment mode."""
    if os.environ.get("OPENZITI"):
        return f"{name.lower()}.miniziti.private"
    if os.environ.get("DOCKER"):
        return name.lower()
    return f"{name.lower()}-service"


def _legacy_kind(name: str) -> str:
    """Legacy convention: a member whose name starts with ``llm-`` is an LLM."""
    return KIND_LLM if name.lower().split("-", 1)[0] == "llm" else KIND_CNN


def backends_from_names(ensemble_members: list[str]) -> list[Backend]:
    """Build backends from the legacy list of bare model names."""
    return [
        Backend(
            name=name,
            kind=_legacy_kind(name),
            url=f"http://{_legacy_host(name)}:{DEFAULT_PORT}{DEFAULT_PATH}",
        )
        for name in ensemble_members
    ]


def backends_from_mappings(entries: list[dict]) -> list[Backend]:
    """Build backends from explicit ``{name, kind, url}`` mappings.

    Lets a mounted YAML / ConfigMap state URLs as fully as the env var does.
    """
    backends = []
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        kind = str(entry.get("kind", KIND_CNN)).strip().lower()
        url = str(entry.get("url", "")).strip()
        if not name or not url:
            raise ValueError(
                f"Invalid backend entry {entry!r}: 'name' and 'url' required"
            )
        if kind not in VALID_KINDS:
            raise ValueError(
                f"Invalid kind {kind!r} for backend {name!r}: expected one of {VALID_KINDS}"
            )
        backends.append(Backend(name=name, kind=kind, url=url))
    return backends


def backends_from_config(config: dict) -> list[Backend]:
    """Build backends from a config dict, preferring its explicit ``backends`` key."""
    entries = config.get("backends")
    if entries:
        return backends_from_mappings(entries)
    return backends_from_names(config.get("ensemble", []))


def resolve_backends(config: dict, prefer_env: bool = True) -> list[Backend]:
    """Return the backends to fan out to.

    ``prefer_env`` is True at startup so docker-compose wins over the image's baked-in
    config. It is False when the config arrives at runtime via ``/change_config``,
    where the caller's intent must win over the deployment default.
    """
    raw = os.environ.get(ENV_VAR)
    if prefer_env and raw and raw.strip():
        backends = parse_backend_spec(raw)
        source = f"${ENV_VAR}"
    else:
        backends = backends_from_config(config)
        source = "ensemble config"

    logging.info(f"Resolved {len(backends)} backend(s) from {source}")
    for backend in backends:
        logging.info(f"  {backend.name} ({backend.kind}) -> {backend.url}")
    return backends
