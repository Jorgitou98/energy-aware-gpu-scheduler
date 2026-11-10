#!/usr/bin/env python3
"""Reproduce Figure 1 from the supplied power-aggregation measurements.

The two board-power observations are averaged. Prediction includes the shared
GPU baseline once and both replicas' residency and rate-dependent increments.
No GPU acquisition or simulation is run by this plotting script.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
import numpy as np

FIGSIZE = (7.05, 2.86)

SINGLE_COLUMN_WIDTH = FIGSIZE[0] / 2

PREVIEW_DPI = 260

INK = "#39434B"

TEAL = "#237F79"

OCHRE = "#AD6C42"

LIGHT = "#D9DEE4"

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
INPUTS = EVIDENCE / "model_examples" / "replica_power.json"

def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "axes.linewidth": .65, "axes.edgecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
    })

def draw_power(fig, power, increments, predicted, measured):
    ax = fig.add_axes([.02, .245, .96, .61])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    base = power["baseline"]
    xp, xm, width = .658, .905, .155
    scale = .70 / max(predicted, measured)

    def card(index, model, y, color):
        x, w, height = .005, .475, .50
        border = tuple(.30 * value + .70 for value in to_rgb(color))
        shape = "round,pad=0.002,rounding_size=0.014"
        patch = FancyBboxPatch((x, y), w, height, boxstyle=shape, facecolor="white", edgecolor=border, lw=.65, clip_on=False)
        ax.add_patch(patch)
        for by, bh, opacity in [(y + .416, .084, .055), (y, .086, .09)]:
            band = Rectangle((x, by), w, bh, facecolor=color, alpha=opacity, edgecolor="none")
            band.set_clip_path(patch)
            ax.add_patch(band)
        rate, e = power["rate"][index], power["residency"][index]
        processing = power["energy_per_request"][index] * rate
        i = index + 1
        entries = [(.452, rf"$c_{i}$: {model}", 7.4, color),
                   (.371, "2 slices, 1200 MHz", 6.9, INK),
                   (.288, rf"$r_{i}={rate:.2f}$ req/s", 7.0, INK),
                   (.205, rf"$e_{{c_{i}}}={e:.2f}$ W", 7.4, INK),
                   (.123, rf"$\alpha_{{c_{i}}}r_{i}={processing:.2f}$ W", 7.4, INK),
                   (.042, rf"$P_{{c_{i}}}(r_{i})={increments[index]:.2f}$ W", 7.4, color)]
        for dy, text, size, ink in entries:
            ax.text(x + .025, y + dy, text, fontsize=size, color=ink, va="center")
        ax.plot([x + .015, x + w - .015], [y + .086, y + .086], color=border, lw=.5)
        return x + w + .005, y + .042

    upper = card(0, "LR-ASPP", .475, TEAL)
    lower = card(1, "DeepLabV3", -.095, OCHRE)
    bottom = 0
    for value, color in [(base, LIGHT), (increments[1], OCHRE), (increments[0], TEAL)]:
        ax.add_patch(Rectangle((xp - width / 2, bottom), width, value * scale, facecolor=color, edgecolor="white", lw=.65))
        bottom += value * scale
    ax.add_patch(Rectangle((xm - width / 2, 0), width, measured * scale, facecolor="#526980", edgecolor="none"))
    center = base * scale / 2
    for dy, label, size in [(.088, "Baseline", 6.1), (.005, r"$P_{gf}^{\mathrm{base}}$", 7.3), (-.075, f"{base:.2f}", 7.1)]:
        ax.text(xp, center + dy, label, ha="center", va="center", fontsize=size)
    for value, y in [(increments[1], (base + increments[1] / 2) * scale),
                     (increments[0], (base + increments[1] + increments[0] / 2) * scale)]:
        ax.text(xp, y, f"{value:.2f}", ha="center", va="center", fontsize=6.9, color="white")
    for x, value, label in [(xp, predicted, "Predicted"), (xm, measured, "Measured")]:
        ax.text(x, value * scale + .035, f"{value:.2f} W", ha="center", va="bottom", fontsize=7.2)
        ax.text(x, -.052, label, ha="center", va="top", fontsize=7.0)
    ax.plot([xp - width / 2 - .01, xm + width / 2 + .01], [0, 0], color=INK, lw=.6)
    ax.text(.781, .90, f"Gap {abs(predicted / measured - 1) * 100:.2f}%", fontsize=8.1, color=TEAL, weight="semibold", ha="center")
    targets = [(xp - width / 2, (base + increments[1] + increments[0] / 2) * scale),
               (xp - width / 2, (base + increments[1] / 2) * scale)]
    for start, end, color in [(upper, targets[0], TEAL), ((lower[0], lower[1] + .42), targets[1], OCHRE)]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=.55, shrinkA=0, shrinkB=2))

def save(fig, directory, name):
    # Crop vertically to visible artwork while preserving the specified page
    # width, as in the supplied generator. Text and curves remain vector PDF.
    fig.canvas.draw()
    initial = fig.get_tightbbox(fig.canvas.get_renderer())
    preview = BytesIO()
    fig.savefig(preview, format="png", dpi=PREVIEW_DPI, bbox_inches=initial, pad_inches=0)
    preview.seek(0)
    pixels = plt.imread(preview)
    occupied = np.any(pixels[:, :, :3] < .995, axis=2)
    ys, xs = np.nonzero(occupied)
    height, width = occupied.shape
    margin = 1 / 72  # One point protects backend-specific glyph extents.
    crop = Bbox.from_extents(
        0,
        initial.y1 - (ys.max() + 1) / height * initial.height - margin,
        fig.get_figwidth(),
        initial.y1 - ys.min() / height * initial.height + margin,
    )
    # Whole preview pixels keep the PDF and PNG physical extents consistent.
    export_height = np.ceil(crop.height * PREVIEW_DPI) / PREVIEW_DPI
    crop = Bbox.from_extents(crop.x0, crop.y0, crop.x1, crop.y0 + export_height)
    fig.savefig(directory / f"{name}.pdf", bbox_inches=crop, pad_inches=0,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(directory / f"{name}.png", dpi=PREVIEW_DPI, bbox_inches=crop, pad_inches=0)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build" / "figures")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    power = json.loads(INPUTS.read_bytes())["inputs"]
    residency = EVIDENCE / "validation" / "residency"
    sources = json.loads((residency / "source.json").read_bytes())
    observed = []
    for name in ("concurrent_0.json", "concurrent_1.json"):
        content = (residency / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != sources["files"][name]["sha256"]:
            raise ValueError(f"Residency acquisition has changed: {name}")
        record = json.loads(content)
        if record["label"] != Path(name).stem or record["rates"] != power["rate"]:
            raise ValueError(f"Residency acquisition has unexpected run conditions: {name}")
        observed.append(fmean(sample["power_W"] for sample in record["samples"]))
    if not np.allclose(observed, power["measured"], rtol=0, atol=1e-12):
        raise ValueError("Raw residency power means do not match the saved comparisons")
    power["measured"] = observed
    increments = np.asarray(power["residency"]) + np.asarray(power["energy_per_request"]) * power["rate"]
    predicted = float(power["baseline"] + increments.sum())
    measured = float(np.mean(power["measured"]))
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        style()
        fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, FIGSIZE[1]))
        draw_power(fig, power, increments, predicted, measured)
        save(fig, args.outdir, "replica_power_aggregation")
    statistics = dict(inputs=power, predicted_power_w=predicted, measured_mean_power_w=measured,
                      replica_increment_w=increments.tolist(), gap_percent=abs(predicted / measured - 1) * 100)
    (args.outdir / "replica_power_aggregation_statistics.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.outdir / "replica_power_aggregation.pdf")


if __name__ == "__main__":
    main()
