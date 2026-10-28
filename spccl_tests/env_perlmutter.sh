# Environment for rs_union_test on Perlmutter, sourced by the run scripts.
# Expects REPO (the torchcomms-sparse checkout) to be set.

NCCLX_LIB="${NCCLX_BUILD_DIR:-$REPO/build/ncclx-union-dev2}/lib"

module load cudatoolkit/12.9
module load nccl/2.24.3                    # libfabric net plugin (needed for multi-node)
export LD_LIBRARY_PATH="$NCCLX_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # dev NCCLX ahead of the module's NCCL

# Perlmutter NCCL/CXI environment, as in the SpCCL sweeps
export CUDA_VISIBLE_DEVICES=3,2,1,0
unset SLURM_MPI_TYPE
unset MPICH_GPU_SUPPORT_ENABLED            # GTL is not linked; MPI only moves host data
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_PROVIDER=cxi
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_OPTIMIZED_MRS=0
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# SpCCL settings (defaults as in the DDP runs the captures come from)
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-16777216}
export NCCL_CCD_FORMAT_MASK=${NCCL_CCD_FORMAT_MASK:-5}
export NCCL_CCD_DENSE_THRESHOLD=${NCCL_CCD_DENSE_THRESHOLD:-0.6}
export NCCL_CCD_DENSE_INTRA_THRESHOLD=${NCCL_CCD_DENSE_INTRA_THRESHOLD:-0.7}
export NCCL_CCD_CHANNELS=${NCCL_CCD_CHANNELS:-64}

# Every task sees all 4 GPUs of its node (so CUDA_VISIBLE_DEVICES above can
# reorder them); the test picks its GPU by local rank. With RUN_TIMEOUT set,
# the step is killed after that many seconds (exit status 124).
run_test() {
  local limit=()
  [[ -n "${RUN_TIMEOUT:-}" ]] && limit=(timeout -k 30 "$RUN_TIMEOUT")
  ${limit[@]+"${limit[@]}"} srun -n "$((SLURM_NNODES * 4))" --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none -c 32 --cpu-bind=cores --mem-bind=local \
    "$REPO/spccl_tests/rs_union_test" "$@"
}
