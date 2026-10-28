#!/bin/bash
# Development build of NCCLX for rs_union_test: separate build dir, A100
# (sm_80) only, and only the kernels the test needs (ReduceScatter sum in
# f16/bf16/f32 and AllGather on Ring, SendRecv, the BitOr AllReduce of the
# union masks and Broadcast on Ring). Other collectives have no
# kernels in this build, so don't point real workloads at it.
#
# Usage: bash spccl_tests/build_ncclx_dev.sh
# Env overrides: BUILDDIR, ONLY_FUNCS (empty = all kernels), NVCC_GENCODE, NCCL_BUILD_JOBS
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
module load cudatoolkit/12.9
module load python

export BUILDDIR="${BUILDDIR:-$REPO/build/ncclx-union-dev2}"
export ONLY_FUNCS="${ONLY_FUNCS-ReduceScatter Sum (f16|bf16|f32) RING *|AllGather RING *|SendRecv|AllReduce BitOr u64 * *|Broadcast RING *}"
export NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_80,code=sm_80}"
export NCCL_BUILD_JOBS="${NCCL_BUILD_JOBS:-16}"

cd "$REPO"
INSTALL_PREFIX=/pscratch/sd/e/egencer/sparsecomms/nccl-prefix \
NCCL_BUILD_SKIP_DEPS=1 \
NCCL_HOME="$REPO/comms/ncclx/stable" \
bash "$REPO/build_ncclx_pm.sh"
