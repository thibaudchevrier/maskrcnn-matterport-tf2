# `lint` runs the pre-commit hooks on every file: the same checks as the git hooks and CI.

.PHONY: install hooks format lint test check

install:
	uv sync --locked --all-extras

hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run pre-commit run --all-files --show-diff-on-failure

test:
	TF_CPP_MIN_LOG_LEVEL=3 uv run pytest -p no:warnings

check: lint test
