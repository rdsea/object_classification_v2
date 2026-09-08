#!/bin/bash
# Pull models into the Ollama container's volume ahead of time.
#
#   ./pull_models.sh                      # pulls $LLM_MODEL (default llava:7b)
#   ./pull_models.sh llava:7b qwen2.5vl:3b
#
# The compose overlay pulls automatically on `up`; this is for warming the cache
# or adding a model to an already-running stack.

set -euo pipefail

SERVICE="${OLLAMA_SERVICE:-ollama}"
MODELS=("$@")

if [ ${#MODELS[@]} -eq 0 ]; then
  IFS=',' read -r -a MODELS <<<"${LLM_MODELS:-${LLM_MODEL:-llava:7b}}"
fi

for model in "${MODELS[@]}"; do
  model="$(echo "$model" | xargs)" # trim
  [ -z "$model" ] && continue
  echo "==> Pulling $model"
  docker compose exec -T "$SERVICE" ollama pull "$model"
done

echo "==> Models available:"
docker compose exec -T "$SERVICE" ollama list
