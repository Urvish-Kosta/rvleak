# Pinned to a specific minor version so that results are reproducible; the
# statistical output depends on Python's RNG, which is stable across patch
# releases but should not be assumed stable across minor ones.
FROM python:3.12-slim

WORKDIR /opt/rvleak
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[dev]"
RUN python -m pytest -q

ENTRYPOINT ["rvleak"]
CMD ["--help"]
