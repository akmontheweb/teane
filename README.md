# myharness

Production-grade, model-agnostic AI Agent Harness with LangGraph orchestration, sandboxed builds, and bulletproof persistence.

## Development

```bash
pip install -e ".[dev]"
make hooks-install
make test
```

The pre-commit hook runs the full 480-test regression pack and blocks any commit that breaks the framework. The hook is enforced locally — to bypass it intentionally (emergencies only), use `git commit --no-verify`.
