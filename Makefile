# Auto-generated Makefile by harness
# Detected project type: Python

.PHONY: build clean test coverage hooks-install release setup

# One-shot bootstrap for a fresh machine. Walks the operator through
# 11 phases: platform / Python / git / sandbox probes → venv → pip
# install → LLM config wizard → harness doctor → optional tools →
# summary. See docs/installation.md §0 for the scripted-install
# overview, or scripts/setup.py --help for CLI flags.
setup:
	@python3 scripts/setup.py $(SETUP_ARGS)

build:
	python -m compileall . 2>/dev/null || python3 -m compileall . 2>/dev/null || echo 'Python compile check skipped'

clean:
	@echo "No clean target configured."

test:
	python -m pytest tests/ -q --tb=short

# Run the pytest pack with coverage measurement. Emits a terminal
# summary plus an HTML report at htmlcov/index.html. No CI gate
# on the coverage number — this target is for local visibility.
# Usage:
#     make coverage           # measure harness/ package, terminal + HTML
#     make coverage SHOW=1    # also open the HTML in a browser when done
coverage:
	@python -m pytest tests/ \
	    --cov=harness \
	    --cov-report=term-missing:skip-covered \
	    --cov-report=html:htmlcov \
	    --cov-report=xml:coverage.xml \
	    -q --tb=short
	@echo ""
	@echo "HTML coverage report: htmlcov/index.html"
	@if [ "$(SHOW)" = "1" ]; then \
	    python -c "import webbrowser, os; webbrowser.open('file://' + os.path.abspath('htmlcov/index.html'))"; \
	fi

hooks-install:
	python -m pre_commit install
	@echo "pre-commit hook installed. Tests will run before every commit."

# Cut a release: verify clean tree, run tests, bump version, update CHANGELOG,
# tag, and push. Usage:
#     make release BUMP=patch    # 1.1.0 -> 1.1.1 (default)
#     make release BUMP=minor    # 1.1.0 -> 1.2.0
#     make release BUMP=major    # 1.1.0 -> 2.0.0
#
# Prompts for confirmation before tagging. Refuses to release with a
# dirty working tree, with a failing test pack, or with no [Unreleased]
# content in CHANGELOG.md.
BUMP ?= patch
release:
	@python scripts/release.py --bump=$(BUMP)
