#!/bin/bash
set -e

cleanup() {
  echo "Interrupted, killing all child processes..."
  kill -- -$$ 2>/dev/null
  wait 2>/dev/null
}
trap cleanup SIGINT SIGTERM

BINARY="cpp/build/obbt_experiment"
MPS_DIR=../sub_collection
OUT_DIR=test_out
N_GPUS=8
N_BATCHES=8

mkdir -p "$OUT_DIR"

pids=()
for batch in $(seq 0 $((N_BATCHES - 1))); do
  gpu_id=$((batch % N_GPUS))
  echo "Launching batch $batch / $N_BATCHES on GPU $gpu_id"
  CUDA_VISIBLE_DEVICES=$gpu_id \
    $BINARY --path "$MPS_DIR" --out-dir "$OUT_DIR" --n-gpus 1 \
    --batch-num "$batch" --n-batches "$N_BATCHES" &
  pids+=($!)
done

echo "All $N_BATCHES batches launched. Waiting..."

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "Batch with PID $pid failed"
    ((failed++))
  fi
done

if [ "$failed" -eq 0 ]; then
  echo "All batches completed successfully"
else
  echo "$failed / $N_BATCHES batches failed"
  exit 1
fi
