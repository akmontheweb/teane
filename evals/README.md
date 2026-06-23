# Teane Eval Harness

A thin scaffolding for measuring harness changes before / after. Walks every
task in `golden_set.yaml`, drives `teane run` against an isolated temp
workspace per task, and emits `results.json` with cost, tokens, cache hit
rate, and per-tool error rates.

## Quickstart

```bash
# Run every task, write results.json
make eval

# Run a single task
python -m evals.run_eval --task fix_off_by_one

# Snapshot the current result as the new baseline
python -m evals.run_eval --output evals/baseline.json

# Compare the latest results.json against baseline.json
python -m evals.compare
```

`compare.py` exits non-zero when any task regresses (passed in baseline,
fails now) so it can gate CI.

## Adding tasks

Edit `golden_set.yaml`. The task shape is documented inline at the top of
that file. For tasks with a starting fixture, drop the seed files under
`evals/fixtures/<task-name>/` and reference them via `fixture_dir`.

`success_check` is a shell command run inside the workspace AFTER the
harness exits. Exit 0 means the task passed. The default rule (harness
exit code 0 = success) applies when `success_check` is omitted.

## What gets recorded

Per task in `results.json`:
- `success` — final verdict (harness exit AND success_check)
- `harness_exit_code`, `check_exit_code`, `wall_clock_s`
- `total_cost_usd`, `llm_call_count`, `tokens_in`, `tokens_out`, `cached_tokens`
- `cache_hit_rate` — derived (audit #26)
- `tool_call_count`, `tool_error_rates` — per tool (audit #15, #27)
- `system_prompt_lines` — bloat metric (audit #8)

The aggregation is read back from the per-session JSONL log files under
`~/.harness/logs/<session>.jsonl` via `harness.metrics.aggregate_session`,
so any future event names auto-flow through with no eval-harness change.
