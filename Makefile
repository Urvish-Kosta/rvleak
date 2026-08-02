.PHONY: help install test lint results figures clean docker

PY ?= python3

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install:  ## install the package and dev dependencies
	$(PY) -m pip install -e ".[dev]"

test:  ## run the test suite
	$(PY) -m pytest -q

lint:  ## static checks
	$(PY) -m ruff check src tests scripts

results:  ## regenerate docs/results.md and assets/*.png
	$(PY) scripts/reproduce.py

quick:  ## fast regeneration for smoke testing
	$(PY) scripts/reproduce.py --quick

demo:  ## short end-to-end demonstration
	$(PY) -m rvleak.cli null
	$(PY) -m rvleak.cli tvla
	$(PY) -m rvleak.cli fullkey -n 600

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

docker:
	docker build -t rvleak .
