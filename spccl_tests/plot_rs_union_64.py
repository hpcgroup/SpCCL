#!/usr/bin/env python3
"""Bar charts of the 64-GPU union-pack RS sweep on the full gradient bucket.

Usage: plot_rs_union_64.py RESULTS_DIR [--out DIR] [--n-ar 63] [--n-ag 32]

Reads RESULTS_DIR/bucket/sweep.csv (rs_union_64gpu.sbatch phase 1: the full
bucket, every captured iteration x channel count). Writes PNG + PDF figures and
summary.csv (the plotted numbers) to RESULTS_DIR/plots unless --out is given:
  fig1_totals          dense RS, SpCCL, and union with the AR mask including
                       extract bitmask + bitmask AR, per channel count
  fig2_nsweep          union RS alone for every N, AG masks and AR mask
  fig3_breakdown       per-phase stacked time of each configuration
CSVs written before the renaming (masks perrank/global, row "mask") load too.

Every phase is timed separately by rs_union_test (its own loop, max over ranks,
mean of the timed iterations), so a configuration's total is a sum of phases,
not one measured run. Numbers are means over the captured iterations.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# Reference palette (light surface).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAR_LIGHT, BAR_DARK = "#6da7ec", "#184f95"

# Phases of a configuration, bottom to top in the stacked chart.
PHASES = [
    ("rs", "Reduce-scatter", BLUE),
    ("exchange", "Mask exchange (bitmask AR or bitmask AG)", ORANGE),
    ("build", "Extract bitmask", AQUA),
]

# Names used before the renaming -> current names.
OLD_MASKS = {"perrank": "ag", "global": "ar"}
OLD_VARIANTS = {"mask": "extract"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8,
})


def load(path, order, pass_counts):
    """{(channels, variant, N, masks): {density: ms}}; density identifies the iteration."""
    t = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["order"]) not in (-1, order) or int(r.get("pass_counts") or 0) not in (-1, pass_counts):
                continue
            variant = OLD_VARIANTS.get(r["variant"], r["variant"])
            key = (int(r["channels"]), variant, int(r["N"]), OLD_MASKS.get(r["masks"], r["masks"]))
            t[key][r["density"]] = float(r["avg_us"]) / 1e3
    return t


def per_iter(t, parts):
    """Per-iteration sum of the given rows."""
    iters = t[parts[0]].keys()
    return np.array([sum(t[p][d] for p in parts) for d in iters])


def configs(c, n_ar, n_ag):
    """(label, {phase: [rows]}) of every configuration at c channels."""
    build = [(c, "extract", 0, "-")]
    return [
        ("Dense", {"rs": [(c, "dense", 0, "-")]}),
        ("SpCCL", {"rs": [(c, "spccl", 1, "-")]}),
        (f"Union\nAR mask\nN={n_ar}",
         {"rs": [(c, "union", n_ar, "ar")], "exchange": [(c, "maskAR", 0, "-")], "build": build}),
        (f"Union\nAG masks\nN={n_ag}",
         {"rs": [(c, "union", n_ag, "ag")], "exchange": [(c, "maskAG", 0, "-")], "build": build}),
    ]


def header(fig, title, subtitle):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=8.5, color=INK2, linespacing=1.4)


def group_labels(ax, centers, labels, y=-0.2):
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, s in zip(centers, labels):
        ax.text(x, y, s, transform=tr, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def fig_totals(t, chans, n_ar, out, rows):
    series = [
        ("Dense reduce-scatter", BLUE, lambda c: [(c, "dense", 0, "-")]),
        ("SpCCL reduce-scatter", ORANGE, lambda c: [(c, "spccl", 1, "-")]),
        (f"Union, AR mask, N={n_ar}: extract bitmask + bitmask AR + RS", AQUA,
         lambda c: [(c, "extract", 0, "-"), (c, "maskAR", 0, "-"), (c, "union", n_ar, "ar")]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.1)
    w = 0.25
    top = 0
    for i, (name, color, parts) in enumerate(series):
        for g, c in enumerate(chans):
            v = per_iter(t, parts(c))
            x = g + (i - 1) * (w + 0.02)
            ax.bar(x, v.mean(), w, color=color, edgecolor=SURFACE, linewidth=0.8, label=name if g == 0 else None)
            ax.vlines(x, v.min(), v.max(), color=INK2, linewidth=0.8)
            ax.text(x, v.max() + 3, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=8, color=INK)
            top = max(top, v.max())
            rows.append(("fig1_totals", c, name, "total", v.mean(), v.min(), v.max()))
    ax.set_xticks(range(len(chans)), [f"{c} channels" for c in chans], fontsize=9.5)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("Time per call (ms)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, handlelength=1.2, borderaxespad=0.3)
    header(fig, "Reduce-scatter of the full 5.8 GiB gradient bucket, 64 GPUs",
           f"Mean over {len(t[(chans[0], 'dense', 0, '-')])} captured iterations (min-max as whiskers). "
           "The union bar is a sum of separately timed phases.")
    save(fig, out, "fig1_totals")


def fig_nsweep(t, chans, ns, out, rows):
    panels = [("ag", "AG masks (each chain ORs its members' masks; needs the bitmask AG)"),
              ("ar", "AR mask (OR of all ranks' masks, one for every chain; needs the bitmask AR)")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.08, hspace=0.62)
    gap = 1.6
    top = max(np.mean(list(t[(c, "union", n, m)].values())) for c in chans for n in ns for m, _ in panels)
    for ax, (masks, title) in zip(axes, panels):
        ticks, labels, centers = [], [], []
        for g, c in enumerate(chans):
            xs = g * (len(ns) + gap) + np.arange(len(ns))
            vals = np.array([np.mean(list(t[(c, "union", n, masks)].values())) for n in ns])
            best = int(vals.argmin())
            colors = [BAR_DARK if i == best else BAR_LIGHT for i in range(len(ns))]
            ax.bar(xs, vals, 0.82, color=colors, edgecolor=SURFACE, linewidth=0.6)
            ax.text(xs[best], vals[best] + top * 0.02, f"N={ns[best]}\n{vals[best]:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK, linespacing=1.1)
            ticks += list(xs)
            labels += [str(n) for n in ns]
            centers.append(xs.mean())
            for n, v in zip(ns, vals):
                rows.append(("fig2_nsweep", c, f"union {masks} N={n}", "rs", v, np.nan, np.nan))
        ax.set_xticks(ticks, labels, fontsize=7)
        ax.set_xlim(-0.8, ticks[-1] + 0.8)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
        ax.set_ylabel("Union RS time (ms)")
        group_labels(ax, centers, [f"{c} channels" for c in chans], y=-0.13)
    axes[0].set_ylim(0, top * 1.15)
    header(fig, "Union-pack reduce-scatter vs chain length N, 64 GPUs, full 5.8 GiB bucket",
           "Reduce-scatter only: extract bitmask and mask exchange excluded. N=1 (plain SpCCL) omitted. "
           "Mean over the captured iterations; darker bar = fastest N in the group.")
    save(fig, out, "fig2_nsweep")


def fig_breakdown(t, chans, n_ar, n_ag, out, rows):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.2)
    w, step, gap = 0.78, 1.12, 0.9
    ticks, labels, centers, bars, x = [], [], [], [], 0
    for c in chans:
        xs = []
        for clabel, phases in configs(c, n_ar, n_ag):
            bottom = 0
            for key, _, color in PHASES:
                if key not in phases:
                    continue
                v = per_iter(t, phases[key]).mean()
                ax.bar(x, v, w, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=0.8)
                rows.append(("fig3_breakdown", c, clabel.replace("\n", " "), key, v, np.nan, np.nan))
                bottom += v
            bars.append((x, per_iter(t, phases["rs"]).mean(), bottom, len(phases) > 1))
            ticks.append(x)
            labels.append(clabel)
            xs.append(x)
            ax.text(x, bottom, f"{bottom:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
            x += step
        centers.append(np.mean(xs))
        x += gap
    top = max(b[2] for b in bars)
    # RS share of the total inside the RS segment of the multi-phase bars, where it fits.
    for x, rs, total, multi in bars:
        if multi and rs > 0.07 * top:
            ax.text(x, rs / 2, f"RS\n{100 * rs / total:.0f}%", ha="center", va="center", fontsize=7.2,
                    color="#ffffff", linespacing=1.1)
    ax.set_xticks(ticks, labels, fontsize=7.2, linespacing=1.15)
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel("Time per call (ms)")
    group_labels(ax, centers, [f"{c} channels" for c in chans], y=-0.17)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in PHASES]
    ax.legend(handles, [name for _, name, _ in PHASES], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              handlelength=1.0, borderaxespad=0.3, columnspacing=1.4)
    header(fig, "Where the time goes: full 5.8 GiB bucket, 64 GPUs",
           "Each phase timed separately and stacked; mean over the captured iterations. "
           "Extract bitmask is the test's own (unoptimized) kernel.")
    save(fig, out, "fig3_breakdown")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-ar", type=int, default=63)
    ap.add_argument("--n-ag", type=int, default=32)
    ap.add_argument("--order", type=int, default=1, help="union packing order to plot (1 tile, 0 element)")
    ap.add_argument("--pass-counts", type=int, default=0, help="union rows to plot: without (0) or with (1) passed tile offsets")
    args = ap.parse_args()
    out = args.out or args.results / "plots"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    t = load(args.results / "bucket" / "sweep.csv", args.order, args.pass_counts)
    chans = sorted({k[0] for k in t})
    ns = sorted({k[2] for k in t if k[1] == "union" and k[2] > 1})
    fig_totals(t, chans, args.n_ar, out, rows)
    fig_nsweep(t, chans, ns, out, rows)
    fig_breakdown(t, chans, args.n_ar, args.n_ag, out, rows)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "group", "config", "phase", "mean_ms", "min_ms", "max_ms"])
        for r in rows:
            w.writerow([*r[:4], *(f"{v:.3f}" if v == v else "" for v in r[4:])])
    print(f"wrote {out / 'summary.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
