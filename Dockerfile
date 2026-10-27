FROM python:3.14.9-slim@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8
WORKDIR /app
COPY pyproject.toml README.md ./
COPY agent_eval ./agent_eval
COPY tests ./tests
RUN pip install --no-cache-dir --root-user-action=ignore ".[test]"
ENTRYPOINT ["python", "-m", "agent_eval.run"]
