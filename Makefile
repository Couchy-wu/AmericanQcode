.PHONY: install install-dev test lint format run-web run-cli-scan clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov httpx ruff mypy

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run-web:
	uvicorn src.web.app:app --host 0.0.0.0 --port 8080 --reload

run-cli-scan:
	qt scan --watchlist default --strategy all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info .pytest_cache/
