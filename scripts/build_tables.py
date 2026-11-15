#!/usr/bin/env python3
"""Recompute Tables 6 and 7, verify publication agreement, then render them."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from efficient_mig.table_results import recompute_tables
METHODS = ('Fixed-frequency', 'Sequential', 'Greedy-C', 'Greedy',
           'Greedy+Beam', 'LeastAllocated', 'MostAllocated')
GROUPS = ('Paired MIG', 'Full GPU', 'Batched MIG', 'Residency')
STATS = ('mean', 'median', 'maximum')


def validate_summary(row):
    if not isinstance(row, dict) or any(key not in row for key in STATS):
        raise ValueError('Incomplete published summary')
    if not all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in STATS):
        raise ValueError('Expected finite published statistics')
    if row['maximum'] < max(row['mean'], row['median']):
        raise ValueError('Maximum is smaller than a published central statistic')


def load_tables():
    data = recompute_tables(ROOT)
    table6, table7 = data['table6'], data['table7']
    if tuple(table6) != GROUPS or tuple(table7) != METHODS:
        raise ValueError('Unexpected published table groups or methods')
    for name, group in table6.items():
        validate_summary(group['power'])
        if name == 'Residency':
            if group['response'] is not None:
                raise ValueError('The final paper does not report Residency response error')
        else:
            validate_summary(group['response'])
    for name, row in table7.items():
        expected_reference = 'Greedy-C' if name in ('LeastAllocated', 'MostAllocated') else 'Exact'
        if row['reference'] != expected_reference or row['attempts'] != 112:
            raise ValueError('Unexpected efficiency reference or attempted cohort')
        for mode in ('predicted', 'measured'):
            validate_summary(row[mode])
    return table6, table7


def render(table6, table7):
    rows6 = []
    for name, group in table6.items():
        values = []
        for metric in ('power', 'response'):
            values.extend(['--'] * 3 if group[metric] is None else
                          [f'{group[metric][stat]:.2f}' for stat in STATS])
        rows6.append(name + ' & ' + ' & '.join(values) + r' \\')
    tex6 = r'''\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.7pt}\renewcommand{\arraystretch}{1.14}
\caption{Absolute errors (\%) by validation group. GPU-power errors are computed from the author-supplied comparison CSV. Response-time errors use archived Paired MIG and Full GPU errors and the 45~s observations from Figure 4 for Batched MIG.}
\label{tab:validation}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{GPU power} & \multicolumn{3}{c}{Response time}\\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
\textbf{Group} & Mean & Median & Max. & Mean & Median & Max.\\
\midrule
''' + '\n'.join(rows6) + '\n' + r'''\bottomrule
\end{tabular*}
\end{table}
'''
    rows7 = []
    for label, row in table7.items():
        if label == 'LeastAllocated':
            rows7.append(r'\midrule')
        values = []
        for mode in ('predicted', 'measured'):
            parts = [format(row[mode][key], '.3f' if label == 'Greedy+Beam' and key == 'mean' else '.2f')
                     for key in STATS]
            values.append(' / '.join(parts))
        rows7.append(label + ' & ' + ' & '.join(values) + r' \\')
    tex7 = r'''\begin{table}[t]
\centering\small
\setlength{\belowcaptionskip}{3pt}
\caption{Predicted and measured percentage losses in utility per joule, relative to Exact for the first five methods and Greedy-C for the Kubernetes baselines.}
\label{tab:GPU_verification_losses}
\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.15}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}
\toprule
\textbf{Method} & \textbf{Predicted (\%)} & \textbf{Measured (\%)}\\
& Mean / Median / Max. & Mean / Median / Max.\\
\midrule
''' + '\n'.join(rows7) + '\n' + r'''\bottomrule
\end{tabular*}
\end{table}
'''
    return tex6, tex7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args()
    outdir = args.outdir or ROOT / 'build/tables'
    recomputed = recompute_tables(ROOT)
    table6, table7 = recomputed['table6'], recomputed['table7']
    tex6, tex7 = render(table6, table7)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, content in [('table6.tex', tex6), ('table7.tex', tex7)]:
        (outdir / name).write_text(content, encoding='utf-8')
    result = {'data_scope': recomputed['audit']['scope'],
              'table6': table6, 'table7': table7, 'recomputation': recomputed['audit']}
    (outdir / 'table_statistics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(outdir.resolve())


if __name__ == '__main__':
    main()
