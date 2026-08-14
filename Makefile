.PHONY: lint format test all

all: format lint test

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check . --fix
	uv run mypy .

format:
	uv run ruff format .

test:
	uv run pytest