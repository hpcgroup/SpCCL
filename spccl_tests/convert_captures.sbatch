#!/bin/bash
# Converts a DDP mask capture into the bitmasks rs_union_test reads, on a
# quarter of a GPU node (shared QOS: 1 GPU + 32 cores; only the CPUs are used).
#   sbatch spccl_tests/convert_captures.sbatch                     newest *_g64_* capture
#   CAPTURE=<mask_captures/JOB_label> sbatch spccl_tests/convert_captures.sbatch
# Converts the captured iterations that are multiples of EVERY (default 10:
# captures happen every 4 iterations, so 20, 40, 60, 80, 100). Output goes to
# spccl_union_data/<capture name>/iter_<k>/ (about 195 MB per rank per iteration).
#SBATCH --account m2404_g
#SBATCH --qos shared
#SBATCH --constraint gpu
#SBATCH --ntasks 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --time 00:30:00
#SBATCH --job-name convert-captures
#SBATCH --output convert_captures_%j.out

set -euo pipefail

ROOT=/pscratch/sd/e/egencer/sparsecomms
CAPTURE=${CAPTURE:-$(ls -d "$ROOT"/Megatron-AxoNN/mask_captures/*_g64_* 2>/dev/null | sort | tail -1)}
EVERY=${EVERY:-10}
JOBS=${JOBS:-16}
if [[ -z "$CAPTURE" || ! -d "$CAPTURE/rank_00000" ]]; then
  echo "no capture found (CAPTURE='$CAPTURE')"
  exit 1
fi
ITERS=$(ls "$CAPTURE/rank_00000" | sed -nE 's/^iter_0*([0-9]+)_.*/\1/p' | sort -un \
        | awk -v e="$EVERY" '$1 % e == 0' | paste -sd, -)
if [[ -z "$ITERS" ]]; then
  echo "no captured iteration in $CAPTURE is a multiple of $EVERY"
  exit 1
fi
OUT=${OUT:-$ROOT/spccl_union_data/$(basename "$CAPTURE")}

module load python   # CPython with numpy (the default python3 here is PyPy)
echo "converting $CAPTURE iterations [$ITERS] -> $OUT with $JOBS workers ($(date +%H:%M:%S))"
python3 "$ROOT/torchcomms-sparse/spccl_tests/convert_captures.py" "$CAPTURE" \
  --iters "$ITERS" --out "$OUT" --jobs "$JOBS"
echo "done ($(date +%H:%M:%S)): $(du -sh "$OUT" | cut -f1) in $OUT"
