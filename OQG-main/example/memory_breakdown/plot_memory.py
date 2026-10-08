#!/usr/bin/env python3
# Plot the memory breakdown results (memory_breakdown.csv) to answer
# reviewer comment O3: stacked absolute-memory bars comparing the CBCA
# edge list (16 level-0 edges) against the traditional full-width edge
# list (64 edges). Two groups: average across all datasets, and GIST.
# Bar height = actual resident memory; stacks = components.
#
# Run with the "draw" conda env:
#   /home/cc/miniconda3/envs/draw/bin/python plot_memory.py
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "memory_breakdown.csv")
OUT_PDF = os.path.join(SCRIPT_DIR, "memory_breakdown.pdf")
OUT_PNG = os.path.join(SCRIPT_DIR, "memory_breakdown.png")

HIGHLIGHT_DS = "gist"

# switch: True = index structures only (exclude raw vectors kept for refinement)
INDEX_ONLY = True  

# ── load ──
data = {}   # (dataset, edge) -> row
order = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        ds, e = r["Dataset"], int(r["EdgeNum"])
        if ds not in order:
            order.append(ds)
        data[(ds, e)] = {k: float(v) if k not in ("Dataset", "opq") else v
                         for k, v in r.items() if k != "Dataset"}

dss = [d for d in order if (d, 16) in data and (d, 64) in data]

# components stacked bottom-up (absolute MB); metadata and LUT are merged
comps_all = [
    (("RawVectorsMB",),        "Raw vectors",     "#bcbddc"),
    (("CodesMB",),             "Quantized codes", "#d62728"),
    (("GraphMB",),             "Graph structure", "#1f77b4"),
    (("MetadataMB", "LUTMB"), "Metadata & LUTs", "#2ca02c"),
]
comps = comps_all[1:] if INDEX_ONLY else comps_all

GB = 1024.0


def comp_values(ds_list, edge):
    """Mean absolute size (GB) of each component over ds_list."""
    return [np.mean([sum(data[(d, edge)][k] for k in keys) for d in ds_list]) / GB
            for keys, _, _ in comps]


groups = [
    (f"Average ({len(dss)} datasets)", dss),
    ("GIST", [HIGHLIGHT_DS]),
]
edges = [64, 16]
edge_label = {64: "w/o CBCA", 16: "w/ CBCA"}

plt.rcParams.update({
    "font.size": 17,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(9.2, 3.0))

W = 0.34
GAP = 0.52          # spacing between the two bars inside a group
centers = np.array([0.0, 1.5])   # group centers

# paddings scale with the tallest bar so INDEX_ONLY mode stays readable
_ymax_raw = max(sum(comp_values(ds, 64)) for _, ds in groups)
top_pad = _ymax_raw * 0.02
bar_label_pad = _ymax_raw * 0.10

for gi, (gname, ds_list) in enumerate(groups):
    for ei, edge in enumerate(edges):
        xpos = centers[gi] + (ei - 0.5) * GAP
        vals = comp_values(ds_list, edge)
        bottom = 0.0
        for (keys, label, color), v in zip(comps, vals):
            ax.bar(xpos, v, W, bottom=bottom, color=color,
                   edgecolor="black", linewidth=0.7,
                   label=label if (gi == 0 and ei == 0) else None)
            bottom += v
        # total on top of each bar
        ax.text(xpos, bottom + top_pad, f"{bottom:.2f}", ha="center",
                fontsize=17, fontweight="bold")
        # edge config under each bar
        ax.text(xpos, -bar_label_pad, edge_label[edge], ha="center", va="top", fontsize=17)

# saving arrow annotation per group
for gi, (gname, ds_list) in enumerate(groups):
    t64 = sum(comp_values(ds_list, 64))
    t16 = sum(comp_values(ds_list, 16))
    sv = 100.0 * (1 - t16 / t64)
    xbar = centers[gi] + 0.5 * GAP
    ax.annotate(f"-{sv:.0f}%",
                xy=(xbar + 0.5 * W, t16 * 0.75),
                xytext=(xbar + 0.28, (t16 + t64) / 2),
                fontsize=19, fontweight="bold", color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.0))

ax.set_ylabel("Memory (GB)" if INDEX_ONLY else "Memory (GB)")
ax.set_xticks(centers)
ax.set_xticklabels([g for g, _ in groups], fontweight="bold")
ax.tick_params(axis="x", pad=30)   # leave room for the per-bar edge labels
ax.set_xlim(-0.75, 2.45)
ymax = max(sum(comp_values(ds, 64)) for _, ds in groups) * 1.18
ax.set_ylim(0, ymax)
ax.set_yticks([0, 2, 4, 6] if ymax > 4 else [0, 1, 2, 3])
ax.grid(axis="y", alpha=0.3, ls="--")
ax.legend(loc="upper left", framealpha=0.95, ncols=2, fontsize=12)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
print(f"saved -> {OUT_PDF}")
print(f"saved -> {OUT_PNG}")

# ── console summary (numbers for the rebuttal text) ──
print(f"\n#datasets = {len(dss)}")
for gname, ds_list in groups:
    print(f"\n[{gname}]")
    for edge in (64, 16):
        vals = comp_values(ds_list, edge)
        total = sum(vals)
        parts = " | ".join(f"{label} {v:.3f}GB ({100*v/total:.1f}%)"
                           for (k, label, c), v in zip(comps, vals))
        print(f"  {edge_label[edge]:>15}: total {total:.3f}GB -> {parts}")
    t64, t16 = sum(comp_values(ds_list, 64)), sum(comp_values(ds_list, 16))
    print(f"  total saved: {100.0 * (1 - t16 / t64):.1f}%")
