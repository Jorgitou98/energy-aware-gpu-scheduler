"""Raster and text comparison of paper references and reproduced PDFs."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pypdf import PdfReader


def compare(repo, out, baseline=None):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((repo / 'figures/figure_manifest.json').read_text())
    results = []
    for row in manifest['figures']:
        n = row['figure']
        paths = {'reference': repo / 'figures' / row['reference'],
                 'reproduced': repo / f'build/figures/figure{n}.pdf'}
        if baseline:
            paths['baseline'] = baseline / f'build/figures/figure{n}.pdf'
        arrays, texts, sizes, hashes = {}, {}, {}, {}
        for label, path in paths.items():
            prefix = out / f'figure{n}_{label}'
            subprocess.run(['pdftoppm', '-f', '1', '-singlefile', '-r', '120',
                            '-png', str(path), str(prefix)], check=True,
                           capture_output=True)
            arrays[label] = np.asarray(Image.open(prefix.with_suffix('.png')).convert('RGB'))
            reader = PdfReader(path)
            assert len(reader.pages) == 1, path
            page = reader.pages[0]
            texts[label] = ' '.join((page.extract_text() or '').split())
            sizes[label] = [float(page.mediabox.width), float(page.mediabox.height)]
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = {}
        for label in paths:
            if label == 'reproduced':
                continue
            a, b = arrays[label], arrays['reproduced']
            metric = {'same_dimensions': a.shape == b.shape,
                      'text_identical': texts[label] == texts['reproduced'],
                      'pdf_identical': hashes[label] == hashes['reproduced']}
            if a.shape == b.shape:
                delta = np.abs(a.astype(float) - b.astype(float))
                metric.update(pixels_identical=bool(np.array_equal(a, b)),
                              changed_pixel_fraction=float(np.any(delta > 0, axis=2).mean()),
                              mean_absolute_channel_difference=float(delta.mean()))
            metrics[label] = metric
        panels = []
        for label in ('reference', 'reproduced'):
            img = Image.fromarray(arrays[label])
            img.thumbnail((1200, 850))
            panel = Image.new('RGB', (1220, 900), 'white')
            ImageDraw.Draw(panel).text((15, 10), f'Figure {n}: {label}', fill='black')
            panel.paste(img, ((1220-img.width)//2, 40))
            panels.append(panel)
        sheet = Image.new('RGB', (2440, 900), '#d0d0d0')
        sheet.paste(panels[0], (0, 0)); sheet.paste(panels[1], (1220, 0))
        sheet.save(out / f'figure{n}_comparison.png')
        results.append({'figure': n, 'name': row['name'], 'sizes_pt': sizes,
                        'hashes': hashes, 'comparisons': metrics})
    report = {'dpi': 120, 'figures': results,
              'note': 'Pixel equality tests rendered content; PDF timestamps may differ. Different fonts/layouts are not numerical failures.'}
    (out / 'figure_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repo', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    compare(args.repo, args.out, args.baseline)
