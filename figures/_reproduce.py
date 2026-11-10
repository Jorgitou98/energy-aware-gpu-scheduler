"""Portable entry points for the numbered publication figures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main(number):
    parser = argparse.ArgumentParser(description=f"Reproduce publication Figure {number}.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "build" / "figures")
    args = parser.parse_args()
    destination = args.outdir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "figure_manifest.json").read_text(encoding="utf-8"))
    entry = next(row for row in manifest["figures"] if row["figure"] == number)
    reference = HERE / entry["reference"]
    if hashlib.sha256(reference.read_bytes()).hexdigest() != entry["reference_sha256"]:
        raise ValueError("The supplied final figure reference has changed")
    if entry["mode"] == "provided_pdf":
        # Export the supplied final paper PDF without rerunning a simulation.
        shutil.copy2(reference, destination / (entry["name"] + ".pdf"))
    else:
        script = HERE / entry["script"]
        subprocess.run([sys.executable, str(script), "--outdir", str(destination)],
                       check=True, cwd=ROOT)
    for extension in ("pdf", "png", "svg"):
        rendered = destination / (entry["name"] + "." + extension)
        if rendered.is_file():
            shutil.copy2(rendered, destination / f"figure{number}.{extension}")
    print(f"Figure {number}: {destination / ('figure' + str(number) + '.pdf')}")
