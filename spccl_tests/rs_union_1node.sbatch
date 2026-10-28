#!/bin/bash
# First union-pack RS test: one Perlmutter node (4 GPUs). Either submit it
# from this directory (cd spccl_tests && sbatch rs_union_1node.sbatch), or run
# it inside an interactive allocation that has the GPUs:
#   salloc -N 1 -C gpu --gpus-per-node 4 -q interactive -A m2404_g -t 30
#   bash spccl_tests/rs_union_1node.sbatch
#
# Runs: iid sanity (f32), then the 16-rank DDP capture (test ranks 0-3 use
# capture ranks 0-3) over the full bucket in f32/f16/bf16 and a 256 MiB
# window. Each run compares dense, SpCCL and union (N=1..4, both orders).
#SBATCH --account m2404_g
#SBATCH --qos regular
#SBATCH --constraint gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --time 00:20:00
#SBATCH --job-name rs-union
#SBATCH --output rs_union_%j.out

set -uo pipefail

# sbatch runs a spooled copy of this script, so fall back to the submit dir.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$HERE/rs_union_test" ]] || HERE="$SLURM_SUBMIT_DIR"
REPO="$(cd "$HERE/.." && pwd)"
DATA="/pscratch/sd/e/egencer/sparsecomms/spccl_union_data/58378110_g16_sparse_noea_bd_r1"
source "$HERE/env_perlmutter.sh"

NTASKS=$((SLURM_NNODES * 4))
run() {
  echo
  echo "=== $*"
  run_test "$@"
}

echo "job $SLURM_JOB_ID on $SLURM_NNODES node(s), $NTASKS ranks; NCCLX from $NCCLX_LIB"
env | grep -E "^NCCL_CCD_|^NCCL_BUFFSIZE" | sort

run --data iid --count 4194304 --density 0.05 --iters 10 --warmup 3
run --data capture --capture "$DATA/iter_000052" --dtype f32 --iters 5 --warmup 2
run --data capture --capture "$DATA/iter_000052" --dtype f16 --iters 5 --warmup 2
run --data capture --capture "$DATA/iter_000052" --dtype bf16 --iters 5 --warmup 2
run --data capture --capture "$DATA/iter_000100" --dtype f32 --iters 5 --warmup 2
run --data capture --capture "$DATA/iter_000052" --dtype f32 --count 16777216 --iters 20 --warmup 5
