#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/madcamp04/FSD"
LOG_DIR="$ROOT/logs"

stop_pid() {
  local name="$1"
  local pid_file="$LOG_DIR/$2.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (PID $pid)..."
      kill "$pid"
    else
      echo "$name PID $pid not running."
    fi
    rm -f "$pid_file"
  else
    echo "$name pid file not found."
  fi
}

stop_pid "LLM (llama-server)" "llama-server"
stop_pid "STT (whisper-server)" "whisper-server"
stop_pid "TTS (gpt-sovits)" "gpt-sovits"

echo "Done."
