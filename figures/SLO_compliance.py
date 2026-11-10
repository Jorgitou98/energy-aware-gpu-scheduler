#!/usr/bin/env python3
"""Render workload and request SLO compliance for the shared-arrival acquisition campaign.

Usage: python SLO_compliance.py --outdir DIRECTORY

Both panels include all 112 workloads. The request percentages use offered
requests, including those without a feasible plan. The mean modeled losses use
only available plans and compare against the nominal Exact reference. The
offline P95/P99 preprocessing time is a shared cost.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from efficient_mig.figure_results import recompute_admission

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "slo-compliance-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#233843"
MUTED = "#49616c"
BASENAME = "SLO_compliance"
POLICIES = ("Nominal", "Variability-aware P95", "Variability-aware P99")
WORKLOAD_CATEGORIES = (("slo_met", "SLO met", "#327763"),
                       ("slo_not_met", "SLO not met", "#f0b281"),
                       ("infeasible", "Infeasible", "#d6dee5"))
REQUEST_LABELS = (("within_slo", r"Response $\leq D_s$"),
                  ("above_slo", r"Response $> D_s$"),
                  ("unserved", "No feasible plan"))


def load_data():
    result = recompute_admission()
    return dict(result["data"], input_audit=result["summary"])


def style_axes(axis, labels, ticks, ylabel):
    axis.set(xlim=(-.53, 2.53), ylim=(0, ticks[-1]), ylabel=ylabel, yticks=ticks)
    axis.set_xticks(range(3), labels, fontsize=6.15)
    axis.tick_params(axis="x", length=0)
    # Keep single-line and wrapped category labels on the publication's rows.
    for tick in axis.xaxis.get_major_ticks():
        tick.set_pad(3.6)
        tick.label1.set_linespacing(1.15)
    axis.tick_params(axis="y", length=2., width=.5, pad=2.6, labelsize=7.5, colors=MUTED)
    axis.yaxis.label.set_size(7.5)
    axis.yaxis.grid(True, color="#d8e1e3", lw=.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(.5)


def add_legend(axis, math_labels=False):
    axis.legend(loc="lower center", bbox_to_anchor=(.5, 1.11652 if math_labels else 1.11806), ncol=3, frameon=False,
                fontsize=6.7, borderaxespad=0, borderpad=0, labelspacing=.3,
                columnspacing=1.15, handletextpad=.42, handlelength=.9, handleheight=.65)


def make_figure(data):
    fig = plt.figure(figsize=(7.223, 2.77), facecolor="white")
    left = fig.add_axes([.067, .175, .437, .637])
    right = fig.add_axes([.607, .175, .368, .637])
    policies = data["policies"]
    total = data["denominator"]
    offered = data["request_denominator"]
    style_axes(left, POLICIES, [0, total / 4, total / 2, total * .75, total], "Workloads")
    style_axes(right, ("Nominal", "Variability-aware\nP95", "Variability-aware\nP99"),
               [0, 25, 50, 75, 100], "Offered requests (%)")

    labels = []
    bottom = [0, 0, 0]
    for field, legend, color in WORKLOAD_CATEGORIES:
        values = [p["counts"][field] for p in policies]
        left.bar(range(3), values, bottom=bottom, width=.78, color=color,
                 edgecolor="white", linewidth=.39, label=legend, zorder=3)
        for x, (count, offset) in enumerate(zip(values, bottom)):
            if count == 0:
                continue
            percent = 100 * count / total
            large = count >= 20
            text = f"{count}\n({percent:.1f}%)" if large else f"{count} ({percent:.1f}%)"
            size = 7.45 if field == "slo_met" else 7.3 if large else 6.7 if field == "infeasible" else 7.05
            label_offset = .5 if field == "slo_not_met" and large else .3 if field == "infeasible" else 0
            left.text(x, offset + count / 2 + label_offset, text, ha="center", va="center",
                      color="white" if field == "slo_met" else INK, fontsize=size,
                      linespacing=1.08 if field == "slo_met" else 1.25, zorder=4)
            labels.append(dict(policy=policies[x]["key"], outcome=field,
                               count=count, percent=percent, text=text))
        bottom = [lo + count for lo, count in zip(bottom, values)]

    bottom = [0., 0., 0.]
    for (field, legend), (_, _, color) in zip(REQUEST_LABELS, WORKLOAD_CATEGORIES):
        values = [100 * p["requests"][field] / offered for p in policies]
        right.bar(range(3), values, bottom=bottom, width=.78, color=color,
                  edgecolor="white", linewidth=.39, label=legend, zorder=3)
        bottom = [lo + value for lo, value in zip(bottom, values)]

    for x, policy in enumerate(policies):
        requests = policy["requests"]
        within = requests["within_slo"]
        right.text(x, 103.15, f"{100 * within / offered:.2f}%", ha="center", va="bottom",
                   fontsize=7.75, fontweight="bold", clip_on=False)
        right.text(x, 47., f"{within:,}\nwithin SLO", ha="center", va="center",
                   fontsize=6.75, fontweight="bold", linespacing=1.15, color="white")
        unserved = requests["unserved"]
        if unserved:
            fraction = 100 * unserved / offered
            right.text(x, 100 - fraction / 2, f"{unserved:,}\n({fraction:.2f}%)",
                       ha="center", va="center", fontsize=6.3, linespacing=.95)
        if "loss" in policy:
            left.text(x, -.1838, f"Loss: {policy['loss']['mean']:.2f}%",
                      transform=left.get_xaxis_transform(), ha="center", va="bottom",
                      fontsize=6.3, color=MUTED, clip_on=False)

    for center, label in ((.2855, "(a) Mean-response SLO compliance"),
                          (.791, "(b) Per-request SLO compliance")):
        fig.text(center, .955, label, ha="center", va="center",
                 fontsize=8.05, fontweight="bold")
    add_legend(left)
    add_legend(right, math_labels=True)
    fig.text(.355, .02486, f"Extra preprocessing time: {data['shared_mc_wall_seconds']:.1f} s (shared)",
             ha="center", va="baseline", fontsize=6.25, color=MUTED)
    fig.text(.791, .03686, f"{offered:,} offered requests per policy across {total} workloads",
             ha="center", va="baseline", fontsize=6.25, color=MUTED)

    return fig, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path(BASENAME))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    data = load_data()
    args.outdir.mkdir(parents=True, exist_ok=True)
    audit_source = {key: value for key, value in data.items() if key != "input_audit"}
    data_digest = hashlib.sha256(json.dumps(audit_source, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.usetex": False,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.autolayout": False, "figure.constrained_layout.use": False,
        "savefig.facecolor": "white", "savefig.transparent": False,
    }):
        fig, labels = make_figure(data)
        for ext in ("pdf", "png"):
            dest = args.outdir / f"{BASENAME}.{ext}"
            extra = {"metadata": {"CreationDate": None, "ModDate": None,
                                   "Creator": "SLO_compliance.py",
                                   "Title": "Workload and per-request SLO compliance"}} if ext == "pdf" else {}
            fig.savefig(dest, dpi=args.dpi, **extra)
            print(dest.resolve())
        plt.close(fig)
    audit = dict(data_sha256=data_digest, matplotlib=matplotlib.__version__,
                 denominator=data["denominator"], request_denominator=data["request_denominator"],
                 repetition=data["repetition"], segment_labels=labels,
                 requests={p["key"]: p["requests"] for p in data["policies"]},
                 loss_annotations={p["key"]: p["loss"]["mean"] for p in data["policies"] if "loss" in p},
                 overhead_seconds=data["shared_mc_wall_seconds"], overhead_shared=True,
                 overhead_shown_once=True, overhead_label_repeated_per_policy=False,
                 input_audit=data["input_audit"])
    (args.outdir / f"{BASENAME}_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
