#!/usr/bin/env python3
# Plot the large-subspace sensitivity results (test_large_m.csv):
# show that once numSubspaces exceeds 256, the 16-bit accumulation
# saturates (SaturationRate ~ 1) and Recall degrades.
#
# Run with the "draw" conda env:
#   /home/cc/miniconda3/envs/draw/bin/python plot_large_m.py
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "test_large_m.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "large_m_sensitivity.pdf")
OUT_PNG = os.path.join(SCRIPT_DIR, "large_m_sensitivity.png")

U16_MAX = 65535
EF_MARK = 200          # representative efSearch for the right panel

# load
rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        if int(r["NumSubspaces"]) > 768 or int(r["NumSubspaces"]) == 384:
            continue
        rows.append({
            "M": int(r["NumSubspaces"]),
            "ef": int(r["efSearch"]),
            "recall": float(r["Recall"]),
            "qps": float(r["QPS"]),
            "sat": float(r["SaturationRate"]),
            "mean": float(r["MeanQDistSum"]),
            "p99": float(r["P99QDistSum"]),
            "max": int(float(r["MaxQDistSum"])),
        })

Ms = sorted(set(r["M"] for r in rows))
by_m = {m: sorted([r for r in rows if r["M"] == m], key=lambda x: x["ef"]) for m in Ms}


def longest_tradeoff_subseq(seq):
    """Longest subsequence (ordered by increasing efSearch) where Recall
    increases and QPS decreases, i.e. clean trade-off points only."""
    n = len(seq)
    if n == 0:
        return seq
    best_len = [1] * n           # LIS length ending at i
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if (seq[j]["recall"] < seq[i]["recall"]
                    and seq[j]["qps"] > seq[i]["qps"]
                    and best_len[j] + 1 > best_len[i]):
                best_len[i] = best_len[j] + 1
                prev[i] = j
    end = max(range(n), key=lambda i: best_len[i])
    out = []
    while end != -1:
        out.append(seq[end])
        end = prev[end]
    return out[::-1]


colors = {128: "#1f77b4", 256: "#2ca02c", 512: "#d62728", 768: "#9467bd"}
markers = {128: "o", 256: "s", 512: "^", 768: "D"}

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

# (a) Recall vs QPS per M (throughput-recall tradeoff),
# keeping only the longest subsequence with increasing Recall / decreasing QPS
for m in Ms:
    d = longest_tradeoff_subseq(by_m[m])
    ax1.plot([r["qps"] for r in d], [r["recall"] for r in d],
             color=colors[m], marker=markers[m], ms=6, lw=1.8,
             markeredgecolor="black", markeredgewidth=0.6,
             label=f"M={m} (sat={d[0]['sat']*100:.0f}%)")
ax1.set_xlabel("Throughput (QPS)")
ax1.set_ylabel("Recall@100")
ax1.set_xscale("log")
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.xaxis.set_minor_formatter(NullFormatter())
ax1.set_xticks([200, 500, 1000, 2000, 5000])
ax1.set_ylim(0.35, 1.02)
ax1.set_yticks([0.4, 0.6, 0.8, 1.0])
ax1.grid(alpha=0.3, ls="--")
ax1.legend(loc="lower right")
ax1.set_title("(a) Recall vs. QPS (GIST)", fontweight="bold", color="black")

# (b) SaturationRate bars + Recall line vs M
x = np.arange(len(Ms))
sat = [by_m[m][0]["sat"] for m in Ms]
rec_mark = []
for m in Ms:
    d = by_m[m]
    exact = [r for r in d if r["ef"] == EF_MARK]
    rec_mark.append(exact[0]["recall"] if exact else max(r["recall"] for r in d))

ax2.bar(x, sat, width=0.55, color="#d62728", alpha=0.75,
        edgecolor="black", linewidth=1.0,
        label="Saturation rate", zorder=2)
for xi, s in zip(x, sat):
    if s < 0.05:
        ax2.text(xi, s + 0.03, f"{s*100:.0f}%", ha="center", fontsize=15,
                 color="black", zorder=4)
    else:
        ax2.text(xi, 0.05, f"{s*100:.0f}%", ha="center", fontsize=15,
                 color="white", zorder=4)
ax2.set_ylabel("Saturation rate")
ax2.set_ylim(0, 1.12)
ax2.set_yticks([0, 0.5, 1.0])
ax2.set_xticks(x)
ax2.set_xticklabels([str(m) for m in Ms])
ax2.set_xlabel("Number of subspaces $M$")

ax2r = ax2.twinx()
ax2r.plot(x, rec_mark, color="#1f77b4", marker="o", ms=8, lw=2,
          markeredgecolor="black", markeredgewidth=0.6,
          label=f"Recall@100 (ef={EF_MARK})", zorder=3)
# for xi, rc in zip(x, rec_mark):
#     ax2r.annotate(f"{rc:.3f}", (xi, rc), textcoords="offset points",
#                   xytext=(0, -18), ha="center", fontsize=15, color="#1f77b4")
ax2r.set_ylabel(f"Recall@100 (ef={EF_MARK})", color="#1f77b4")
ax2r.tick_params(axis="y", labelcolor="#1f77b4")
ax2r.set_ylim(0.1, 1.05)
ax2r.set_yticks([0.2, 0.6, 1.0])

# u16 limit reference: 255*M crosses 65535 between M=256 and M=512
ax2.axvline(x=1.5, color="gray", ls=":", lw=1.5)
ax2.text(1.5, 1.05, r"$255 \times M > 65535$ (uint16 limit)", ha="center", fontsize=15, color="gray")

h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2r.get_legend_handles_labels()
# lower-left corner is free (zero-height bars there); keep clear of the
# descending recall line and the tall bars on the right
ax2.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.02, 0.10),
           framealpha=0.95)
ax2.set_title("(b) Saturation rate and Recall vs. $M$ (GIST)", fontweight="bold", color="black")

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
print(f"saved -> {OUT_PATH}")
print(f"saved -> {OUT_PNG}")

# console summary
print(f"\n{'M':>6} {'sat%':>8} {'meanSum':>10} {'p99Sum':>10} {'maxSum':>8} "
      f"{'rec@' + str(EF_MARK):>9} {'bestRec':>8}")
for m, s, rc in zip(Ms, sat, rec_mark):
    d = by_m[m][0]
    best = max(r["recall"] for r in by_m[m])
    print(f"{m:>6} {s*100:>7.2f}% {d['mean']:>10.0f} {d['p99']:>10.0f} "
          f"{d['max']:>8} {rc:>9.3f} {best:>8.3f}")
print(f"\nu16 limit = {U16_MAX}; sums exceed it once 255*M > 65535 (M > 257).")
