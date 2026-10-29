#!/bin/bash
# Sweep of union-pack RS on the DDP mask captures, over channel counts x
# capture iterations x dtypes x N x packing order x union mask kind, with all
# mask exchanges timed (bitmask allgather, COO allgather, COO per-rank
# broadcasts, BitOr allreduce). Run inside an allocation, e.g. 4 nodes:
#   salloc -N 4 -C gpu --gpus-per-node 4 -q interactive -A m2404_g -t 60
#   bash spccl_tests/sweep_capture.sh
#
# Every knob below can be overridden from the environment, e.g. a quick smoke
# test:  CHANNELS=64 NS=1,15 ITERS=3 bash spccl_tests/sweep_capture.sh
# or the full grid:  CHANNELS="8 16 32 64" CAPTURE_ITERS="52 100" NS= ORDERS=1,0 ...
# SIZES (MiB of sendbuff per rank) selects a window from the start of the
# bucket instead of the whole bucket.
#
# Each channel count sets NCCL_CCD_CHANNELS and, as in the DDP runs,
# NCCL_MAX_NCHANNELS (CAP_DENSE=0 leaves dense at NCCL's own choice).
#
# Output: $OUT/sweep.csv (one row per measurement) and one log per srun.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/spccl_tests/env_perlmutter.sh"

CAPTURE_DIR=${CAPTURE_DIR:-"/pscratch/sd/e/egencer/sparsecomms/spccl_union_data/58378110_g16_sparse_noea_bd_r1"}
CAPTURE_ITERS=${CAPTURE_ITERS:-"52"}
CHANNELS=${CHANNELS:-"16 32 64"}
DTYPES=${DTYPES:-"f32"}
SIZES=${SIZES:-""}                                  # empty: the whole bucket
CAPTURE_COUNT=${CAPTURE_COUNT:-""}                  # recvcount per rank; empty: numel / nranks (the whole bucket)
NS=${NS-"1,2,4,8,12,14,15,16"}                      # empty: 1..nranks
ORDERS=${ORDERS:-"1"}                               # 1 = tile (SPOP) order, 0 = element order
PASS_COUNTS=${PASS_COUNTS:-"0"}                     # 1 = union hops send U's tile offsets with the values
UNION_MASKS=${UNION_MASKS:-"ag,ar"}                  # ag = AG masks, ar = AR mask
EXCHANGES=${EXCHANGES:-"bitmask,coo-ag,coo-bcast,orreduce"}
VARIANTS=${VARIANTS:-"dense,spccl,mask,union"}
COUNT_EXCHANGE=${COUNT_EXCHANGE:-"mpi"}             # how COO exchanges share counts: mpi or nccl (nccl hangs)
ITERS=${ITERS:-5}
WARMUP=${WARMUP:-2}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}                     # seconds per srun
CAP_DENSE=${CAP_DENSE:-1}
OUT=${OUT:-"$REPO/spccl_tests/results/sweep_capture_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$OUT"
CSV="$OUT/sweep.csv"
echo "sweep_capture: $SLURM_NNODES node(s), $((SLURM_NNODES * 4)) ranks -> $OUT"
echo "  capture $CAPTURE_DIR iterations [$CAPTURE_ITERS] sizes [${SIZES:-bucket}] count [${CAPTURE_COUNT:-bucket}] dtypes [$DTYPES]"
echo "  channels [$CHANNELS] N [${NS:-1..P}] orders [$ORDERS] union masks [$UNION_MASKS] pass counts [$PASS_COUNTS]"
echo "  variants [$VARIANTS] exchanges [$EXCHANGES] counts via $COUNT_EXCHANGE"
echo "  iters $ITERS warmup $WARMUP cap_dense $CAP_DENSE timeout ${RUN_TIMEOUT}s"
env | grep -E "^NCCL_CCD_|^NCCL_BUFFSIZE" | sort > "$OUT/env.txt"

for c in $CHANNELS; do
  for it in $CAPTURE_ITERS; do
    for dtype in $DTYPES; do
      export NCCL_CCD_CHANNELS=$c
      if (( CAP_DENSE )); then export NCCL_MAX_NCHANNELS=$c; else unset NCCL_MAX_NCHANNELS; fi
      cap="$CAPTURE_DIR/$(printf 'iter_%06d' "$it")"
      log="$OUT/run_${c}ch_iter${it}_${dtype}.log"
      echo "=== channels=$c iteration=$it dtype=$dtype ($(date +%H:%M:%S))"
      args=(--data capture --capture "$cap" --dtype "$dtype" --order "$ORDERS" --pass-counts "$PASS_COUNTS" --variants "$VARIANTS"
            --union-masks "$UNION_MASKS" --exchange "$EXCHANGES" --count-exchange "$COUNT_EXCHANGE"
            --iters "$ITERS" --warmup "$WARMUP" --csv)
      [[ -n "$NS" ]] && args+=(--n "$NS")
      [[ -n "$SIZES" ]] && args+=(--size "$SIZES")
      [[ -n "$CAPTURE_COUNT" ]] && args+=(--count "$CAPTURE_COUNT")
      # The CSV rows are rank 0's lines that start with a number; keep one header.
      # A hung run is killed after RUN_TIMEOUT seconds; its rows so far are kept.
      run_test "${args[@]}" 2>&1 \
        | tee "$log" | grep --line-buffered -E '^(nranks,|[0-9]+,)' \
        | { [[ -s "$CSV" ]] && grep --line-buffered -v '^nranks,' || cat; } >> "$CSV"
      (( PIPESTATUS[0] == 124 )) && echo "!!! timed out after ${RUN_TIMEOUT}s: $log"
      if grep -q -E "FAIL|CCD UNION|NCCL error|CUDA error" "$log"; then
        echo "!!! problems in $log:"; grep -E "FAIL|CCD UNION|NCCL error|CUDA error" "$log" | head -5
      fi
    done
  done
done
echo "done ($(date +%H:%M:%S)): $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
