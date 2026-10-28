#!/bin/bash
# Builds rs_union_test against an NCCLX build (default: the dev build from
# build_ncclx_dev.sh).
#
# Usage: bash spccl_tests/build_test.sh    (env: NCCLX_BUILD_DIR, OUT)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
module load cudatoolkit/12.9
NCCLX="${NCCLX_BUILD_DIR:-$HERE/../build/ncclx-union-dev2}"
OUT="${OUT:-$HERE/rs_union_test}"

nvcc -O2 -std=c++17 -arch=sm_80 -o "$OUT" "$HERE/rs_union_test.cu" \
  -I"$NCCLX/include" -I"$MPICH_DIR/include" \
  -L"$NCCLX/lib" -lnccl -Xlinker -rpath="$NCCLX/lib" \
  -L"$MPICH_DIR/lib" -lmpi
echo "built $OUT against $NCCLX"
