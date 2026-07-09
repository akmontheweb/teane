## Agent Standards & Conventions

### Patch Quality
- Every SEARCH block must be an exact, unique substring of the file being patched. Test search blocks mentally before outputting.
- REPLACE_BLOCK should change only the minimum lines needed; never rewrite entire files.
- CREATE_FILE must include the full file content with proper imports at the top.
- Never generate patches that touch more than 3 files in a single response without explicit approval.

### Move Semantics — Two-Sided Fixes
When a diagnostic says the fix is a **move** or **relocation** — e.g. "move X to the top-level conftest", "this belongs at Y instead of Z", "should be defined in the parent module", "extract this into a shared utility" — the fix has TWO halves and BOTH must land in the SAME round:
- **Destination:** CREATE_FILE (new path) or REPLACE_BLOCK (existing path) to introduce the content at the target.
- **Source:** DELETE_BLOCK / REWRITE_FILE / DELETE_FILE to remove the content from the original path.

Only doing the destination leaves the source in violation and the diagnostic re-fires next round, wasting a repair cycle. If the source file becomes empty (or contains only the moved content) after removal, DELETE_FILE it entirely rather than leaving a stub.

Example — pytest error: *"Defining `pytest_plugins` in a non-top-level conftest is no longer supported; move it to the top-level conftest"*:
- Right: CREATE_FILE `/conftest.py` with the plugin line **AND** REWRITE_FILE or DELETE_FILE `backend/tests/conftest.py`.
- Wrong: CREATE_FILE `/conftest.py` alone — the offending line still exists at the old path and pytest fails identically.

### Error Handling
- All new functions must include try/except blocks for external calls (API, file I/O, database).
- Return meaningful error messages, not raw exception strings.
- Use custom exception classes where appropriate; avoid bare `except:`.

### Modularity
- Each file should have a single, clear responsibility.
- Extract reusable logic into utility functions/classes.
- Keep functions under 50 lines. If longer, refactor into sub-functions.
- Use dependency injection rather than hardcoded imports where practical.

### Type Safety
- Python: All function signatures must have type hints (parameters and return types).
- TypeScript/JavaScript: Use TypeScript interfaces/types; avoid `any`.
- Go: Every exported function must have a doc comment.

### Testing
- When creating new modules, suggest the test file structure.
- Test files should mirror the source structure (e.g., `src/auth.py` → `tests/test_auth.py`).
- Use descriptive test names that explain the scenario being tested.

### Documentation
- Every new class and public function must have a docstring.
- Docstrings must describe parameters, return values, and raised exceptions.
- Module-level docstrings should explain the module's purpose.

### Security
- Never include API keys, tokens, passwords, or secrets in generated code.
- Use environment variables for configuration; reference them via `os.environ.get()`.
- Validate all external inputs before processing.
- Sanitize data before logging; never log credentials.

### Performance
- Avoid O(n²) patterns; prefer dict/set lookups over list scans.
- Cache expensive computations where appropriate.
- Use async I/O for network and file operations.
