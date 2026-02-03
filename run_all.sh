#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/madcamp04/FSD"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

echo "Starting LLM (llama-server)..."
nohup "$ROOT/third_party/llama.cpp/build/bin/llama-server" \
  -m "$ROOT/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf" \
  --host 0.0.0.0 --port 8000 \
  > "$LOG_DIR/llama-server.log" 2>&1 &
echo $! > "$LOG_DIR/llama-server.pid"

echo "Starting STT (whisper-server)..."
nohup "$ROOT/third_party/whisper.cpp/build/bin/whisper-server" \
  -m "$ROOT/models/ggml-large-v3.bin" \
  --host 0.0.0.0 --port 8081 -l ko \
  > "$LOG_DIR/whisper-server.log" 2>&1 &
echo $! > "$LOG_DIR/whisper-server.pid"

echo "Starting TTS (GPT-SoVITS api_v2.py)..."
cd "$ROOT/GPT-SoVITS"
nohup python api_v2.py -a 0.0.0.0 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml \
  > "$LOG_DIR/gpt-sovits.log" 2>&1 &
echo $! > "$LOG_DIR/gpt-sovits.pid"

echo "All services started."
echo "Logs: $LOG_DIR"
