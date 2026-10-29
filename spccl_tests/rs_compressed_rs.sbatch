#!/bin/bash
# Compressed RS vs union-pack RS on a DDP capture, full bucket, every converted
# iteration x CHANNELS (default 8 16 32 64), on P = 4 x nodes GPUs:
#   dense          ncclReduceScatter
#   spccl          ncclReduceScatterSparse
#   compressed-rs  extract bitmask -> bitmask AR -> count_elems -> compress -> dense RS -> decompress
#   union-e2e      extract bitmask -> bitmask AR -> union-pack RS with the AR mask, N = P-1 and P,
#                  without and with the tile offsets passed along with the values (PASS_COUNTS=0,1),
#                  in tile order (ORDERS 1) and as {value, index} pairs (ORDERS 2, N = P only)
#   union          with UNION_MASKS=growing_union (add union to VARIANTS): the growing
#                  union RS, no masks, N = P only (needs a test binary built with it:
#                  RS_UNION_TEST and NCCLX_BUILD_DIR, see env_perlmutter.sh)
# The pipelines are timed with an event after each phase (see rs_union_test.cu).
# NCCL_CCD_CHANNELS = NCCL_MAX_NCHANNELS = the channel count, and the test's
# own kernels launch that many blocks of 512 threads.
#
# The node count comes from sbatch (16 below); the capture defaults to the
# newest conversion taken at the same GPU count (*_g<P>_*), otherwise set
# CAPTURE_DIR (test rank r then uses capture rank r). Run compressed_rs_smoke.sh first.
#   sbatch spccl_tests/rs_compressed_rs.sbatch                  64 GPUs
#   sbatch --nodes 4 spccl_tests/rs_compressed_rs.sbatch        16 GPUs
#   CAPTURE_DIR=<spccl_union_data/...> sbatch --nodes 1 spccl_tests/rs_compressed_rs.sbatch
#SBATCH --account m2404_g
#SBATCH --qos regular
#SBATCH --constraint gpu&hbm40g
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --time 00:30:00
#SBATCH --job-name rs-compressed
#SBATCH --output rs_compressed_rs_%j.out

set -uo pipefail

TESTS=${SPCCL_TESTS:-/pscratch/sd/e/egencer/sparsecomms/torchcomms-sparse/spccl_tests}
DATA_ROOT=/pscratch/sd/e/egencer/sparsecomms/spccl_union_data
P=$((SLURM_NNODES * 4))
CAPTURE_DIR=${CAPTURE_DIR:-$(ls -d "$DATA_ROOT"/*_g${P}_* 2>/dev/null | sort | tail -1)}
CONVERTED=$(ls "$CAPTURE_DIR"/iter_*/manifest.txt 2>/dev/null | sed -nE 's#.*/iter_0*([0-9]+)/manifest.txt#\1#p' | sort -n | paste -sd' ' -)
if [[ -z "$CAPTURE_DIR" || -z "$CONVERTED" ]]; then
  echo "no converted capture for $P ranks (CAPTURE_DIR='$CAPTURE_DIR'); set CAPTURE_DIR"
  exit 1
fi

export CAPTURE_DIR
export CAPTURE_ITERS=${CAPTURE_ITERS:-$CONVERTED}
export CHANNELS=${CHANNELS:-"8 16 32 64"}
export VARIANTS=${VARIANTS:-dense,spccl,compressed-rs,union-e2e}
export UNION_MASKS=${UNION_MASKS:-ar}
export NS=${NS:-$((P - 1)),$P}
export ORDERS=${ORDERS:-1,2}
export PASS_COUNTS=${PASS_COUNTS:-0,1}
export ITERS=${ITERS:-5}
export WARMUP=${WARMUP:-2}
export OUT="$TESTS/results/rs_compressed_rs_g${P}_${SLURM_JOB_ID:-nojob}"
echo "job ${SLURM_JOB_ID:-?}: $P ranks, capture $CAPTURE_DIR (iterations $CAPTURE_ITERS) -> $OUT"
bash "$TESTS/sweep_capture.sh"
