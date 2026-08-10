#!/usr/bin/env bash
# Regenerate the demo -> BC chain against the 2026-08-08 PLA-recentred physics.
#
# Writes to *_pla directories so the old-physics artifacts (server_full_5000,
# imitation_v1, bc_policy_v1) survive as the optimistic-physics baseline that
# v3/v4 were measured on. Stops before the SAC fine-tune: BC quality is the
# gate, and the fine-tune is a separate 4.5 h commit.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

suffix="${1:-pla}"
workers="${2:-20}"
shard_episodes="${3:-25}"
demos_dir="artifacts/local_segments/server_full_5000_${suffix}"
imitation_dir="artifacts/local_segments/imitation_${suffix}"
bc_dir="artifacts/local_segments/bc_policy_${suffix}"
mkdir -p "$demos_dir"

echo "== 1/3 generating demos ($workers workers) =="
pids=()
for ((worker=0; worker<workers; worker++)); do
  label="$(printf '%02d' "$worker")"
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u -m train.generate_segment_demos \
      --geometries-per-kind 50 --conditions-per-geometry 20 \
      --out "$demos_dir/worker-$label.npz" \
      --shard-episodes "$shard_episodes" \
      --workers "$workers" --worker-index "$worker" --resume \
      > "$demos_dir/worker-$label.log" 2>&1 &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [[ "$fail" -ne 0 ]]; then echo "a demo worker failed; see logs" >&2; exit 1; fi
echo "demos done"

echo "== 2/3 preparing geometry-disjoint dataset =="
.venv/bin/python -u -m train.prepare_imitation_dataset \
  --shard-dir "$demos_dir" \
  --dataset artifacts/local_segments/balanced_dataset.json \
  --out-dir "$imitation_dir"

echo "== 3/3 cloning BC =="
.venv/bin/python -u -m train.behavior_clone \
  --data-dir "$imitation_dir" --out-dir "$bc_dir" --device cuda:0

echo "== chain complete: $bc_dir =="
