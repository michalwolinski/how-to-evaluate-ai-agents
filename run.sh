#!/usr/bin/env bash
# ./run.sh test                         unit tests and the scripted-agent checks, no API key needed
# ./run.sh eval [args...]               python -m agent_eval.run inside the container (see --help)
# ./run.sh report runs/<run-dir>        summary of one run, recomputed from its rows
# ./run.sh regrade runs/<run-dir> [args]  re-judge saved episodes, no agent calls (see --help)
# ./run.sh charts                       the article's SVG charts, drawn from runs/ into charts/
#
# The API key is passed to the container for this one command only. If ANTHROPIC_API_KEY is not
# set, on macOS it is read from the keychain item "ANTHROPIC_API_KEY".
set -euo pipefail
cd "$(dirname "$0")"
docker build -q -t agent-eval:local . >/dev/null
mkdir -p runs

case "${1:-}" in
  test)
    docker run --rm --entrypoint python agent-eval:local -m pytest -q
    ;;
  report)
    docker run --rm -v "$PWD/runs:/app/runs" --entrypoint python agent-eval:local -m agent_eval.report "/app/$2"
    ;;
  charts)
    mkdir -p charts
    docker run --rm -v "$PWD/runs:/app/runs:ro" -v "$PWD/charts:/app/charts" --entrypoint python agent-eval:local -m agent_eval.charts charts
    ;;
  eval|regrade)
    module=$([ "$1" = eval ] && echo run || echo regrade)
    shift
    if [ -z "${ANTHROPIC_API_KEY:-}" ] && command -v security >/dev/null; then
      ANTHROPIC_API_KEY="$(security find-generic-password -s ANTHROPIC_API_KEY -w 2>/dev/null || true)"
    fi
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
    docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/runs:/app/runs" --entrypoint python agent-eval:local -m "agent_eval.$module" "$@"
    ;;
  *)
    sed -n '2,9p' "$0"; exit 1
    ;;
esac
