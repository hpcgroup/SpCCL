#!/bin/bash
# Correctness check of the Compressed RS and union-e2e pipelines and of the
# growing union RS on one node (4 GPUs), to run before rs_compressed_rs.sbatch,
# inside an interactive allocation:
#   salloc -N 1 -C gpu --gpus-per-node 4 -q debug -A m2404_g -t 20
#   bash spccl_tests/compressed_rs_smoke.sh
# (for a test binary built against another NCCLX, set RS_UNION_TEST and
# NCCLX_BUILD_DIR, see env_perlmutter.sh).
# iid data with shard lengths that are not multiples of 64 (or of the
# 4096-element tile), every dtype and a few kernel grids, then the full bucket
# of the 64-rank capture (test ranks 0-3 use capture ranks 0-3), plus one
# union run with the AG masks. Union runs go without and with the tile offsets
# passed along with the values, and with {value, index} pairs (order 2, fp32,
# N = 4 only); the iid ones with NCCL_CCD_UNION_VERIFY=1, so receivers also
# count U and trap if the offsets or pair counts they got differ. The growing
# union RS (NCCL_CCD_UNION_MASK=growing_union and growing_union_v2, fp32, N = 4)
# runs with the iid data, where density 0.3 makes slices switch to dense on the
# way, and on the capture; with VERIFY=1 growing_union's receivers check each
# pair count against the popcount of the received mask (v2 has no such check).
# Every result is compared bitwise with dense ncclReduceScatter.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/env_perlmutter.sh"
CAPTURE=${CAPTURE:-$(ls -d /pscratch/sd/e/egencer/sparsecomms/spccl_union_data/*_g64_*/iter_000060 | tail -1)}
LOG=${LOG:-$HERE/results/compressed_rs_smoke_${SLURM_JOB_ID:-nojob}.log}
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

common=(--variants dense,compressed-rs,union-e2e --n 3,4 --order 0,1,2 --pass-counts 0,1 --iters 2 --warmup 1)
export NCCL_CCD_UNION_VERIFY=1
for dtype in f32 f16 bf16; do
  for grid in 1 8 64; do
    for count in 100 1000003; do
      echo "=== iid dtype=$dtype grid=$grid count=$count" | tee -a "$LOG"
      run_test --data iid --density 0.01,0.3 --overlap 0.3 --count "$count" --dtype "$dtype" --grid "$grid" \
        "${common[@]}" >> "$LOG" 2>&1
    done
  done
done
# Growing union needs recvcount to be a multiple of 4 (16-byte aligned slices);
# 1000004 is still not a multiple of 64 or of the 4096-element tile.
for grid in 1 64; do
  for count in 100 1000004; do
    echo "=== iid growing union grid=$grid count=$count" | tee -a "$LOG"
    run_test --data iid --density 0.01,0.3 --overlap 0.3 --count "$count" --dtype f32 --grid "$grid" \
      --variants dense,union --union-masks growing_union,growing_union_v2 --n 4 --iters 2 --warmup 1 >> "$LOG" 2>&1
  done
done
echo "=== iid union with the AG and AR masks" | tee -a "$LOG"
run_test --data iid --density 0.05 --overlap 0.3 --count 1000003 --variants dense,union --union-masks ag,ar \
  --n 1,2,3,4 --order 0,1,2 --pass-counts 0,1 --iters 2 --warmup 1 >> "$LOG" 2>&1
echo "=== capture $CAPTURE (full bucket), growing union (VERIFY=1) and v2" | tee -a "$LOG"
run_test --data capture --capture "$CAPTURE" --variants dense,union --union-masks growing_union,growing_union_v2 \
  --n 4 --iters 2 --warmup 1 >> "$LOG" 2>&1
export NCCL_CCD_UNION_VERIFY=0
echo "=== capture $CAPTURE (full bucket)" | tee -a "$LOG"
run_test --data capture --capture "$CAPTURE" --variants dense,spccl,compressed-rs,union-e2e,union \
  --union-masks growing_union,growing_union_v2 --n 3,4 --order 1,2 --pass-counts 0,1 --iters 3 --warmup 1 >> "$LOG" 2>&1

pass=$(grep -c -E '^((compressedRS|unionE2E)\.total|union) .* PASS' "$LOG")
fail=$(grep -c -E 'FAIL|CCD UNION|NCCL error|CUDA error|compressed-rs:.*changed' "$LOG")
echo "results passed: $pass, problems: $fail (log: $LOG)"
grep -E 'FAIL|CCD UNION|NCCL error|CUDA error' "$LOG" | head -5
