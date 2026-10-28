#!/usr/bin/env python3
"""Convert DDP pruned-mask captures into raw bitmasks for rs_union_test.

Input: a capture directory written by Megatron-AxoNN's pruned_mask_capture.py
(rank_<r>/iter_<k>_bucket_<off>_numel_<n>.npz, delta-encoded COO indices),
with any number of ranks and one bucket per iteration.

Output, per requested iteration:
  <out>/iter_<k>/rank_<r>.bits    bit (e % 8) of byte (e / 8) set iff element e
                                  is nonzero (the flat layout the kernels use)
  <out>/iter_<k>/manifest.txt     "key value" lines: nranks, numel, iteration,
                                  source, and density_<r> per rank

Usage: convert_captures.py CAPTURE_DIR --iters 20,40,60 --out OUT_DIR [--jobs N]
"""

import argparse
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def load_capture(path):
    with np.load(path, allow_pickle=False) as c:
        if str(c["encoding"]) != "delta_coo" or str(c["bit_order"]) != "flat_c_order":
            raise ValueError(f"{path}: unsupported encoding")
        indices = np.cumsum(c["index_deltas"], dtype=np.int64)
        return indices, int(c["numel"]), int(c["rank"])


def to_bits(indices, numel):
    """Little-endian packed bitmask of the (sorted, unique) indices."""
    packed = np.zeros((numel + 7) // 8, dtype=np.uint8)
    if indices.size:
        byte = indices >> 3
        bit = np.left_shift(np.uint8(1), (indices & 7).astype(np.uint8))
        starts = np.flatnonzero(np.r_[True, byte[1:] != byte[:-1]])
        packed[byte[starts]] = np.bitwise_or.reduceat(bit, starts)
    return packed


def convert_rank(task):
    """Converts one rank's capture of one iteration; returns (rank, nnz, numel)."""
    rank_dir, r, iteration, out = task
    files = sorted(Path(rank_dir).glob(f"iter_{iteration:06d}_bucket_*_numel_*.npz"))
    if len(files) != 1:
        raise RuntimeError(f"{rank_dir}: expected one bucket for iteration {iteration}, found {len(files)}")
    indices, numel, rank = load_capture(files[0])
    if rank != r:
        raise RuntimeError(f"{files[0]}: rank {rank} in a directory for rank {r}")
    if indices.size > 1 and not np.all(indices[1:] > indices[:-1]):
        raise RuntimeError(f"{files[0]}: indices are not strictly increasing")
    path = Path(out) / f"rank_{r:05d}.bits"
    tmp = path.with_suffix(".tmp")
    to_bits(indices, numel).tofile(tmp)
    os.replace(tmp, path)
    return r, int(indices.size), numel


def convert_iteration(capture_dir, rank_dirs, iteration, out_dir, jobs):
    out = out_dir / f"iter_{iteration:06d}"
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(str(d), r, iteration, str(out)) for r, d in enumerate(rank_dirs)]
    if jobs > 1:
        with Pool(jobs) as pool:
            results = pool.map(convert_rank, tasks)
    else:
        results = [convert_rank(t) for t in tasks]
    numels = {n for _, _, n in results}
    if len(numels) != 1:
        sys.exit(f"iteration {iteration}: ranks disagree on numel: {sorted(numels)}")
    numel = numels.pop()
    densities = [nnz / numel for _, nnz, _ in sorted(results)]
    with open(out / "manifest.txt", "w") as f:
        f.write(f"nranks {len(rank_dirs)}\n")
        f.write(f"numel {numel}\n")
        f.write(f"iteration {iteration}\n")
        f.write(f"source {capture_dir.resolve()}\n")
        for r, d in enumerate(densities):
            f.write(f"density_{r} {d:.6f}\n")
    print(f"wrote {out}: {len(rank_dirs)} ranks, numel {numel}, "
          f"density {min(densities):.5f}-{max(densities):.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--iters", required=True, help="comma-separated iterations, e.g. 20,40,60")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (one rank file each)")
    args = ap.parse_args()

    rank_dirs = sorted(d for d in args.capture_dir.iterdir() if re.fullmatch(r"rank_\d+", d.name))
    if not rank_dirs:
        sys.exit(f"no rank_* directories in {args.capture_dir}")
    ranks = [int(d.name.split("_")[1]) for d in rank_dirs]
    if ranks != list(range(len(ranks))):
        sys.exit(f"rank directories are not 0..{len(ranks) - 1}: {ranks}")

    for it in (int(x) for x in args.iters.split(",") if x):
        convert_iteration(args.capture_dir, rank_dirs, it, args.out, args.jobs)


if __name__ == "__main__":
    main()
