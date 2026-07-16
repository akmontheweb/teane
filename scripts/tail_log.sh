#!/usr/bin/env bash
#
# tail_log.sh — follow a Teane session log in a readable form.
#
# The harness writes one JSON object per line to
#   ${HARNESS_LOG_DIR:-~/.harness/logs}/<session_id>.jsonl
# and maintains a `latest.jsonl` symlink pointing at the current run, so by
# default this follows whatever session is running right now.
#
# Usage:
#   scripts/tail_log.sh                  # follow latest.jsonl
#   scripts/tail_log.sh <session_id>     # follow a specific session
#   scripts/tail_log.sh /path/to.jsonl   # follow an explicit file
#   scripts/tail_log.sh -l WARNING       # only show WARNING and above
#   scripts/tail_log.sh -r               # raw JSON lines (no jq formatting)
#   scripts/tail_log.sh -n 200 <id>      # start with the last 200 lines
#
# Env:
#   HARNESS_LOG_DIR   overrides the log directory (default ~/.harness/logs)
#
set -euo pipefail

LOG_DIR="${HARNESS_LOG_DIR:-$HOME/.harness/logs}"
MIN_LEVEL=""
RAW=0
TAIL_LINES=0          # 0 → start empty (follow new lines only)
TARGET=""

die() { printf 'tail_log: %s\n' "$1" >&2; exit 1; }

# ---- parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--level)  MIN_LEVEL="${2:-}"; shift 2 ;;
    -r|--raw)    RAW=1; shift ;;
    -n|--lines)  TAIL_LINES="${2:-0}"; shift 2 ;;
    -h|--help)   sed -n '2,26p' "$0"; exit 0 ;;
    -*)          die "unknown option: $1" ;;
    *)           TARGET="$1"; shift ;;
  esac
done

# ---- resolve the file to follow ------------------------------------------
if [[ -z "$TARGET" ]]; then
  FILE="$LOG_DIR/latest.jsonl"
elif [[ -f "$TARGET" ]]; then
  FILE="$TARGET"                       # explicit path
else
  FILE="$LOG_DIR/${TARGET%.jsonl}.jsonl"   # treat as a session id
fi

# Wait (briefly) for the file/symlink to appear — handy when you launch the
# tail a beat before `teane run` has created the session log.
if [[ ! -e "$FILE" ]]; then
  printf 'tail_log: waiting for %s ...\n' "$FILE" >&2
  for _ in $(seq 1 30); do
    [[ -e "$FILE" ]] && break
    sleep 1
  done
  [[ -e "$FILE" ]] || die "no such log file: $FILE (is a session running? is HARNESS_LOG_DIR right?)"
fi

printf 'tail_log: following %s\n' "$(readlink -f "$FILE" 2>/dev/null || echo "$FILE")" >&2

# ---- raw mode / no jq -----------------------------------------------------
if [[ "$RAW" -eq 1 ]] || ! command -v jq >/dev/null 2>&1; then
  [[ "$RAW" -eq 1 ]] || printf 'tail_log: jq not found — showing raw JSON (install jq for pretty output)\n' >&2
  exec tail -n "$TAIL_LINES" -F "$FILE"
fi

# ---- level gate (numeric) -------------------------------------------------
lvl_num() { case "$1" in DEBUG) echo 10;; INFO) echo 20;; WARNING|WARN) echo 30;;
                         ERROR) echo 40;; CRITICAL) echo 50;; *) echo 0;; esac; }
MIN_NUM=0
[[ -n "$MIN_LEVEL" ]] && MIN_NUM=$(lvl_num "$(printf '%s' "$MIN_LEVEL" | tr '[:lower:]' '[:upper:]')")

# Colorize only when stdout is a terminal (not when piped to a file/pager).
USE_COLOR=0
[[ -t 1 ]] && USE_COLOR=1

# ---- pretty stream --------------------------------------------------------
# jq builds "HH:MM:SS LEVEL [event] k=v k=v"  (or the plain message line).
# A tolerant fallback prints any non-JSON line verbatim (e.g. tracebacks).
# awk then colorizes by level. Level names are padded to 8 for alignment.
tail -n "$TAIL_LINES" -F "$FILE" 2>/dev/null \
  | jq -Rr --argjson min "$MIN_NUM" '
      . as $raw
      | (try fromjson catch null) as $o
      | if $o == null then $raw            # not JSON → pass through untouched
        else
          ($o.level // "INFO") as $lvl
          | ({DEBUG:10,INFO:20,WARNING:30,WARN:30,ERROR:40,CRITICAL:50}[$lvl] // 0) as $ln
          | if $ln < $min then empty
            else
              (($o.ts // "") | (if length >= 19 then .[11:19] else . end)) as $t
              | if ($o.event // "") != "" then
                  ($o | del(.ts,.level,.logger,.msg,.event)) as $rest
                  | ($rest | to_entries | map("\(.key)=\(.value|tostring)") | join(" ")) as $kv
                  | "\($t) \($lvl) [\($o.event)] \($kv)"
                else
                  "\($t) \($lvl) \($o.msg // "")"
                end
            end
        end
    ' \
  | awk -v color="$USE_COLOR" '
      {
        if (!color) { print; next }
        lvl=$2; c="";
        if (lvl=="ERROR"||lvl=="CRITICAL") c="\033[31m";      # red
        else if (lvl=="WARNING"||lvl=="WARN") c="\033[33m";   # yellow
        else if (lvl=="INFO") c="\033[36m";                   # cyan
        else if (lvl=="DEBUG") c="\033[90m";                  # grey
        if (c!="") printf "%s%s\033[0m\n", c, $0;
        else print $0;
      }' 2>/dev/null || true
