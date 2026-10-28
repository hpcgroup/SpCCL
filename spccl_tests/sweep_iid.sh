#!/bin/bash
# Sweep of union-pack RS on iid data (the worst case: independent supports),
# over channel counts x sendbuff sizes x sparsities x N x packing order.
# Run inside an allocation, e.g. 4 nodes:
#   salloc -N 4 -C gpu --gpus-per-node 4 -q interactive -A m2404_g -t 60
#   bash spccl_tests/sweep_iid.sh
#
# Every knob below can be overridden from the environment, e.g. a quick smoke
# test:  CHANNELS=64 SIZES=64 DENSITIES=0.01 ITERS=5 bash spccl_tests/sweep_iid.sh
#
# Each channel count sets NCCL_CCD_CHANNELS and, as in the DDP runs,
# NCCL_MAX_NCHANNELS (so the dense baseline gets the same cap; CAP_DENSE=0
# leaves dense at NCCL's own choice). Channels are fixed at communicator
# init, so each channel count is one srun; sizes and sparsities are swept
# inside it.
#
# Output: $OUT/sweep.csv (one row per variant/config) and $OUT/run_<C>ch_<dtype>.log

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/spccl_tests/env_perlmutter.sh"

CHANNELS=${CHANNELS:-"8 16 32 64"}
SIZES=${SIZES:-"64,128,256,512"}                    # MiB of sendbuff per rank
DENSITIES=${DENSITIES:-"0.2,0.1,0.05,0.01,0.0001"}  # sparsity 80, 90, 95, 99, 99.99 %
DTYPES=${DTYPES:-"f32"}
NS=${NS:-""}                                        # empty: 1..nranks
ORDERS=${ORDERS:-"1,0"}
PASS_COUNTS=${PASS_COUNTS:-"0"}
UNION_MASKS=${UNION_MASKS:-"ag,ar"}                  # ag = AG masks, ar = AR mask
EXCHANGES=${EXCHANGES:-"bitmask,coo-ag,coo-bcast,orreduce"}
VARIANTS=${VARIANTS:-"dense,spccl,mask,union"}
COUNT_EXCHANGE=${COUNT_EXCHANGE:-"mpi"}             # how COO exchanges share counts: mpi or nccl (nccl hangs)
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}                    # seconds per srun
ITERS=${ITERS:-20}
WARMUP=${WARMUP:-5}
CAP_DENSE=${CAP_DENSE:-1}
OUT=${OUT:-"$REPO/spccl_tests/results/sweep_iid_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$OUT"
CSV="$OUT/sweep.csv"
echo "sweep_iid: $SLURM_NNODES node(s), $((SLURM_NNODES * 4)) ranks -> $OUT"
echo "  channels [$CHANNELS] sizes [$SIZES] MiB densities [$DENSITIES] dtypes [$DTYPES]"
echo "  N [${NS:-1..P}] orders [$ORDERS] pass counts [$PASS_COUNTS] union masks [$UNION_MASKS] exchanges [$EXCHANGES]"
echo "  variants [$VARIANTS] counts via $COUNT_EXCHANGE iters $ITERS warmup $WARMUP"
echo "  cap_dense $CAP_DENSE timeout ${RUN_TIMEOUT}s"
env | grep -E "^NCCL_CCD_|^NCCL_BUFFSIZE" | sort > "$OUT/env.txt"

for c in $CHANNELS; do
  for dtype in $DTYPES; do
    export NCCL_CCD_CHANNELS=$c
    if (( CAP_DENSE )); then export NCCL_MAX_NCHANNELS=$c; else unset NCCL_MAX_NCHANNELS; fi
    log="$OUT/run_${c}ch_${dtype}.log"
    echo "=== channels=$c dtype=$dtype ($(date +%H:%M:%S))"
    args=(--data iid --dtype "$dtype" --size "$SIZES" --density "$DENSITIES" --order "$ORDERS" --pass-counts "$PASS_COUNTS"
          --variants "$VARIANTS" --union-masks "$UNION_MASKS" --exchange "$EXCHANGES"
          --count-exchange "$COUNT_EXCHANGE"
          --iters "$ITERS" --warmup "$WARMUP" --csv)
    [[ -n "$NS" ]] && args+=(--n "$NS")
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
echo "done ($(date +%H:%M:%S)): $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
