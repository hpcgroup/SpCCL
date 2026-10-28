#!/bin/bash
# Union-pack RS on the 64-GPU DDP capture (last of three jobs):
#   1. capture:  NNODES_LIST=16 QOS=regular bash Megatron-AxoNN/breakdown_experiment_scaling/launch_all.sh
#   2. convert:  sbatch spccl_tests/convert_captures.sbatch
#   3. this:     sbatch spccl_tests/rs_union_64gpu.sbatch
# Uses the newest *_g64_* conversion (CAPTURE_DIR=... picks another one).
#
# Phase 1, every converted iteration x CHANNELS: the full 5.8 GiB bucket with
# dense, SpCCL, the bitmask AG (AG masks), the bitmask AR (AR mask) and union
# with both.
# Phase 2, one iteration at 64 channels: the first 2 GiB of the bucket with all
# four exchanges, including COO. On 40 GB GPUs the full bucket fits the first
# two exchanges (~27 GB per GPU: every rank holds all 64 masks) but not COO too
# (~47 GB).
#SBATCH --account m2404_g
#SBATCH --qos regular
#SBATCH --constraint gpu&hbm40g
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --time 00:45:00
#SBATCH --job-name rs-union-64
#SBATCH --output rs_union_64_%j.out

set -uo pipefail

TESTS=${SPCCL_TESTS:-/pscratch/sd/e/egencer/sparsecomms/torchcomms-sparse/spccl_tests}
DATA_ROOT=/pscratch/sd/e/egencer/sparsecomms/spccl_union_data
CAPTURE_DIR=${CAPTURE_DIR:-$(ls -d "$DATA_ROOT"/*_g64_* 2>/dev/null | sort | tail -1)}
CONVERTED=$(ls "$CAPTURE_DIR"/iter_*/manifest.txt 2>/dev/null | sed -nE 's#.*/iter_0*([0-9]+)/manifest.txt#\1#p' | sort -n | paste -sd' ' -)
if [[ -z "$CAPTURE_DIR" || -z "$CONVERTED" ]]; then
  echo "no converted 64-rank capture found (CAPTURE_DIR='$CAPTURE_DIR'); run convert_captures.sbatch first"
  exit 1
fi
read -ra converted <<< "$CONVERTED"

export CAPTURE_DIR
export NS=${NS-"1,2,4,8,16,32,48,56,60,62,63,64"}
export ORDERS=${ORDERS:-1}
export ITERS=${ITERS:-5}
export WARMUP=${WARMUP:-2}
RESULTS="$TESTS/results/rs_union_64_${SLURM_JOB_ID:-nojob}"
echo "job ${SLURM_JOB_ID:-?}: capture $CAPTURE_DIR (iterations $CONVERTED) -> $RESULTS"

# 1) Full bucket, every converted iteration
CAPTURE_ITERS=${CAPTURE_ITERS:-$CONVERTED} CHANNELS=${CHANNELS:-"16 32 64"} \
  EXCHANGES=bitmask,orreduce OUT="$RESULTS/bucket" bash "$TESTS/sweep_capture.sh"

# 2) First 2 GiB of the bucket, all four exchanges, middle iteration, 64 channels
CAPTURE_ITERS=${WINDOW_ITERS:-${converted[$(( ${#converted[@]} / 2 ))]}} CHANNELS=${WINDOW_CHANNELS:-64} \
  SIZES=2048 EXCHANGES=bitmask,coo-ag,coo-bcast,orreduce OUT="$RESULTS/window2g" bash "$TESTS/sweep_capture.sh"
