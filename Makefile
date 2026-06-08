# Auto-generated Makefile by harness
# Detected project type: Python

.PHONY: build clean test hooks-install

build:
	python -m compileall . 2>/dev/null || python3 -m compileall . 2>/dev/null || echo 'Python compile check skipped'

clean:
	@echo "No clean target configured."

test:
	python -m pytest tests/ -q --tb=short

hooks-install:
	python -m pre_commit install
	@echo "pre-commit hook installed. Tests will run before every commit."
