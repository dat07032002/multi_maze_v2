#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

workers="${1:-20}"
episodes_per_shard="${2:-25}"
run_dir="artifacts/local_segments/server_full_5000"
pid_file="$run_dir/pids.txt"
mkdir -p "$run_dir"

if [[ -s "$pid_file" ]]; then
  while read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "teacher worker $pid is already running" >&2
      exit 2
    fi
  done < "$pid_file"
fi

: > "$pid_file"
for ((worker=0; worker<workers; worker++)); do
  label="$(printf '%02d' "$worker")"
  log="$run_dir/worker-$label.log"
  output="$run_dir/worker-$label.npz"
  nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u -m train.generate_segment_demos \
      --geometries-per-kind 50 \
      --conditions-per-geometry 20 \
      --out "$output" \
      --shard-episodes "$episodes_per_shard" \
      --workers "$workers" \
      --worker-index "$worker" \
      --resume > "$log" 2>&1 &
  echo "$!" >> "$pid_file"
done

echo "started $workers workers"
echo "PID file: $project_dir/$pid_file"
echo "logs: $project_dir/$run_dir/worker-XX.log"
