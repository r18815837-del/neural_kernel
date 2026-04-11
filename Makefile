PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

.PHONY: install install-dev test run clean

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: install

test:
	$(PYTEST) -q

run:
	$(PYTHON) examples/01_linear_regression.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf .pytest_cache build dist *.egg-info
