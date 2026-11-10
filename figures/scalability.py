#!/usr/bin/env python3
"""Figure 8: planner scaling on ten workloads per size.

All 160 observations are ordinary Python dictionaries in DATA below. There
are no compressed/encoded data, external inputs, network requests, or LaTeX
requirements. Install Matplotlib and run this script with Python 3.10+:

    python -m pip install matplotlib
    python scalability.py

Optional: --outdir figures --dpi 300

The outputs are a vector PDF, PNG, caption text, and a layout/audit JSON.
Lines are arithmetic means of full planning time. Bands are interquartile
ranges across workloads, not confidence intervals. Every observation is
shown as a small point. Labels report mean paired optimality loss only for
the heuristic methods. Ten workloads are evaluated per size; these are not
ten timing repetitions of the same input. Larger fleets use the model,
not physical GPU replay.

Loss is computed as 100 * (1 - ratio / exact_ratio) for each matched case.
The renderer checks the four methods, workload coverage and paired references
before aggregating the observations and exporting the figure.

Typography: DejaVu Sans; Matplotlib defaults are reset, and PDF fonts are
embedded TrueType. No system LaTeX installation or custom fonts are needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox

SIZES = (16, 32, 64, 128)
SEEDS = (317, 331, 347, 359, 373, 389, 401, 419, 433, 449)
EXPECTED = {f"s{seed}_n{n}" for seed in SEEDS for n in SIZES}
METHODS = ("exact_cold", "exact_initialized", "greedy_beam", "greedy")
STYLES = {
    "exact_cold": ("Exact (cold)", "#82608f", "^", "--"),
    "exact_initialized": ("Exact (initialized)", "#a14e31", "s", "-"),
    "greedy_beam": ("Greedy+Beam", "#23747a", "o", "-"),
    "greedy": ("Greedy", "#458745", "+", "--"),
}

SOURCE_SHA256 = '898855b5f0f0f2784942f59603e99f61fa63a8a07ddb4559e234fe455c5ac27f'
DATA_SHA256 = '8b4bde5bc3582cdd89b71b1b8f6beac230ee72d81e8d9fb293b30e93573f5b0b'

# Ordinary, uncompressed observations from the compact analysis.
DATA = {'exact_cold': [{'case': 's317_n16',
                 'services': 16,
                 'time_s': 8.583950645290315,
                 'ratio': 0.33483347182584167,
                 'exact_ratio': 0.33483347182584167,
                 'loss_percent': 0.0},
                {'case': 's331_n16',
                 'services': 16,
                 'time_s': 4.4672164125368,
                 'ratio': 0.33249959311933036,
                 'exact_ratio': 0.33249959311933036,
                 'loss_percent': 0.0},
                {'case': 's347_n16',
                 'services': 16,
                 'time_s': 5.657192927785218,
                 'ratio': 0.33691022607935533,
                 'exact_ratio': 0.33691022607935533,
                 'loss_percent': 0.0},
                {'case': 's359_n16',
                 'services': 16,
                 'time_s': 5.659415904432535,
                 'ratio': 0.31574179045192874,
                 'exact_ratio': 0.31574179045192874,
                 'loss_percent': 0.0},
                {'case': 's373_n16',
                 'services': 16,
                 'time_s': 7.446956302970648,
                 'ratio': 0.3261417556392365,
                 'exact_ratio': 0.3261417556392365,
                 'loss_percent': 0.0},
                {'case': 's389_n16',
                 'services': 16,
                 'time_s': 6.756177112460136,
                 'ratio': 0.33084554322959425,
                 'exact_ratio': 0.33084554322959425,
                 'loss_percent': 0.0},
                {'case': 's401_n16',
                 'services': 16,
                 'time_s': 4.747844653204083,
                 'ratio': 0.3454327924145958,
                 'exact_ratio': 0.3454327924145958,
                 'loss_percent': 0.0},
                {'case': 's419_n16',
                 'services': 16,
                 'time_s': 6.627030385658145,
                 'ratio': 0.30542451783728614,
                 'exact_ratio': 0.30542451783728614,
                 'loss_percent': 0.0},
                {'case': 's433_n16',
                 'services': 16,
                 'time_s': 7.724528845399618,
                 'ratio': 0.35439729991815344,
                 'exact_ratio': 0.35439729991815344,
                 'loss_percent': 0.0},
                {'case': 's449_n16',
                 'services': 16,
                 'time_s': 12.287905875593424,
                 'ratio': 0.3273896156191229,
                 'exact_ratio': 0.3273896156191229,
                 'loss_percent': 0.0},
                {'case': 's317_n32',
                 'services': 32,
                 'time_s': 40.22271458618343,
                 'ratio': 0.34952949472549,
                 'exact_ratio': 0.34952949472549,
                 'loss_percent': 0.0},
                {'case': 's331_n32',
                 'services': 32,
                 'time_s': 51.908520920202136,
                 'ratio': 0.3305103125018311,
                 'exact_ratio': 0.3305103125018311,
                 'loss_percent': 0.0},
                {'case': 's347_n32',
                 'services': 32,
                 'time_s': 23.07568679191172,
                 'ratio': 0.3235184166476103,
                 'exact_ratio': 0.3235184166476103,
                 'loss_percent': 0.0},
                {'case': 's359_n32',
                 'services': 32,
                 'time_s': 62.349665420129895,
                 'ratio': 0.32779708700776017,
                 'exact_ratio': 0.32779708700776017,
                 'loss_percent': 0.0},
                {'case': 's373_n32',
                 'services': 32,
                 'time_s': 41.161417776718736,
                 'ratio': 0.33853474442505505,
                 'exact_ratio': 0.33853474442505505,
                 'loss_percent': 0.0},
                {'case': 's389_n32',
                 'services': 32,
                 'time_s': 17.52225429750979,
                 'ratio': 0.32399690372871226,
                 'exact_ratio': 0.32399690372871226,
                 'loss_percent': 0.0},
                {'case': 's401_n32',
                 'services': 32,
                 'time_s': 37.60803730040789,
                 'ratio': 0.33980311014844333,
                 'exact_ratio': 0.33980311014844333,
                 'loss_percent': 0.0},
                {'case': 's419_n32',
                 'services': 32,
                 'time_s': 123.6056247446686,
                 'ratio': 0.3209678398163322,
                 'exact_ratio': 0.3209678398163322,
                 'loss_percent': 0.0},
                {'case': 's433_n32',
                 'services': 32,
                 'time_s': 22.365834079682827,
                 'ratio': 0.3465182955767683,
                 'exact_ratio': 0.3465182955767683,
                 'loss_percent': 0.0},
                {'case': 's449_n32',
                 'services': 32,
                 'time_s': 31.477010060101748,
                 'ratio': 0.3412590958704517,
                 'exact_ratio': 0.3412590958704517,
                 'loss_percent': 0.0},
                {'case': 's317_n64',
                 'services': 64,
                 'time_s': 188.94480735622346,
                 'ratio': 0.34139159767288935,
                 'exact_ratio': 0.34139159767288935,
                 'loss_percent': 0.0},
                {'case': 's331_n64',
                 'services': 64,
                 'time_s': 67.40290083549917,
                 'ratio': 0.32569174458955075,
                 'exact_ratio': 0.32569174458955075,
                 'loss_percent': 0.0},
                {'case': 's347_n64',
                 'services': 64,
                 'time_s': 201.06563411839306,
                 'ratio': 0.3307670809583909,
                 'exact_ratio': 0.3307670809583909,
                 'loss_percent': 0.0},
                {'case': 's359_n64',
                 'services': 64,
                 'time_s': 64.83005530945957,
                 'ratio': 0.33643720092746726,
                 'exact_ratio': 0.33643720092746726,
                 'loss_percent': 0.0},
                {'case': 's373_n64',
                 'services': 64,
                 'time_s': 177.6589526552707,
                 'ratio': 0.3336222969845459,
                 'exact_ratio': 0.3336222969845459,
                 'loss_percent': 0.0},
                {'case': 's389_n64',
                 'services': 64,
                 'time_s': 166.48069442994893,
                 'ratio': 0.3218470342002856,
                 'exact_ratio': 0.3218470342002856,
                 'loss_percent': 0.0},
                {'case': 's401_n64',
                 'services': 64,
                 'time_s': 123.45011407136917,
                 'ratio': 0.3317926906408422,
                 'exact_ratio': 0.3317926906408422,
                 'loss_percent': 0.0},
                {'case': 's419_n64',
                 'services': 64,
                 'time_s': 186.93915392644703,
                 'ratio': 0.3224064766737971,
                 'exact_ratio': 0.3224064766737971,
                 'loss_percent': 0.0},
                {'case': 's433_n64',
                 'services': 64,
                 'time_s': 153.26363803260028,
                 'ratio': 0.33516290487518474,
                 'exact_ratio': 0.33516290487518474,
                 'loss_percent': 0.0},
                {'case': 's449_n64',
                 'services': 64,
                 'time_s': 165.7132191043347,
                 'ratio': 0.3431540730957014,
                 'exact_ratio': 0.3431540730957014,
                 'loss_percent': 0.0},
                {'case': 's317_n128',
                 'services': 128,
                 'time_s': 1018.8599296156317,
                 'ratio': 0.339139069476943,
                 'exact_ratio': 0.339139069476943,
                 'loss_percent': 0.0},
                {'case': 's331_n128',
                 'services': 128,
                 'time_s': 569.8632135540247,
                 'ratio': 0.32620019131790645,
                 'exact_ratio': 0.32620019131790645,
                 'loss_percent': 0.0},
                {'case': 's347_n128',
                 'services': 128,
                 'time_s': 1332.400435641408,
                 'ratio': 0.3265298544837086,
                 'exact_ratio': 0.3265298544837086,
                 'loss_percent': 0.0},
                {'case': 's359_n128',
                 'services': 128,
                 'time_s': 1593.785470444709,
                 'ratio': 0.3300698947880742,
                 'exact_ratio': 0.3300698947880742,
                 'loss_percent': 0.0},
                {'case': 's373_n128',
                 'services': 128,
                 'time_s': 671.1011504735798,
                 'ratio': 0.3297017259849892,
                 'exact_ratio': 0.3297017259849892,
                 'loss_percent': 0.0},
                {'case': 's389_n128',
                 'services': 128,
                 'time_s': 2732.620459968224,
                 'ratio': 0.3226167293239902,
                 'exact_ratio': 0.3226167293239902,
                 'loss_percent': 0.0},
                {'case': 's401_n128',
                 'services': 128,
                 'time_s': 1869.6984628550708,
                 'ratio': 0.3295106185839803,
                 'exact_ratio': 0.3295106185839803,
                 'loss_percent': 0.0},
                {'case': 's419_n128',
                 'services': 128,
                 'time_s': 1461.4862181581557,
                 'ratio': 0.3267674873785109,
                 'exact_ratio': 0.3267674873785109,
                 'loss_percent': 0.0},
                {'case': 's433_n128',
                 'services': 128,
                 'time_s': 778.6753052007407,
                 'ratio': 0.3341173986064007,
                 'exact_ratio': 0.3341173986064007,
                 'loss_percent': 0.0},
                {'case': 's449_n128',
                 'services': 128,
                 'time_s': 1052.320065703243,
                 'ratio': 0.337118662279818,
                 'exact_ratio': 0.3371186622805257,
                 'loss_percent': 2.0993207172637085e-10}],
 'exact_initialized': [{'case': 's317_n16',
                        'services': 16,
                        'time_s': 16.753495105542243,
                        'ratio': 0.33483347182584167,
                        'exact_ratio': 0.33483347182584167,
                        'loss_percent': 0.0},
                       {'case': 's331_n16',
                        'services': 16,
                        'time_s': 14.242031474597752,
                        'ratio': 0.33249959311933036,
                        'exact_ratio': 0.33249959311933036,
                        'loss_percent': 0.0},
                       {'case': 's347_n16',
                        'services': 16,
                        'time_s': 15.94136083777994,
                        'ratio': 0.33691022607935533,
                        'exact_ratio': 0.33691022607935533,
                        'loss_percent': 0.0},
                       {'case': 's359_n16',
                        'services': 16,
                        'time_s': 16.878228291869164,
                        'ratio': 0.31574179045192874,
                        'exact_ratio': 0.31574179045192874,
                        'loss_percent': 0.0},
                       {'case': 's373_n16',
                        'services': 16,
                        'time_s': 17.01017752662301,
                        'ratio': 0.3261417556392365,
                        'exact_ratio': 0.3261417556392365,
                        'loss_percent': 0.0},
                       {'case': 's389_n16',
                        'services': 16,
                        'time_s': 17.673565808683634,
                        'ratio': 0.33084554322959425,
                        'exact_ratio': 0.33084554322959425,
                        'loss_percent': 0.0},
                       {'case': 's401_n16',
                        'services': 16,
                        'time_s': 15.430626090615988,
                        'ratio': 0.3454327924145958,
                        'exact_ratio': 0.3454327924145958,
                        'loss_percent': 0.0},
                       {'case': 's419_n16',
                        'services': 16,
                        'time_s': 15.950430814176798,
                        'ratio': 0.30542451783728614,
                        'exact_ratio': 0.30542451783728614,
                        'loss_percent': 0.0},
                       {'case': 's433_n16',
                        'services': 16,
                        'time_s': 14.624593317508698,
                        'ratio': 0.35439729991815344,
                        'exact_ratio': 0.35439729991815344,
                        'loss_percent': 0.0},
                       {'case': 's449_n16',
                        'services': 16,
                        'time_s': 14.887233573943377,
                        'ratio': 0.3273896156191229,
                        'exact_ratio': 0.3273896156191229,
                        'loss_percent': 0.0},
                       {'case': 's317_n32',
                        'services': 32,
                        'time_s': 18.983204784803092,
                        'ratio': 0.34952949472549,
                        'exact_ratio': 0.34952949472549,
                        'loss_percent': 0.0},
                       {'case': 's331_n32',
                        'services': 32,
                        'time_s': 31.767056765034795,
                        'ratio': 0.3305103125018311,
                        'exact_ratio': 0.3305103125018311,
                        'loss_percent': 0.0},
                       {'case': 's347_n32',
                        'services': 32,
                        'time_s': 14.303299561142921,
                        'ratio': 0.3235184166476103,
                        'exact_ratio': 0.3235184166476103,
                        'loss_percent': 0.0},
                       {'case': 's359_n32',
                        'services': 32,
                        'time_s': 37.077011777088046,
                        'ratio': 0.32779708700776017,
                        'exact_ratio': 0.32779708700776017,
                        'loss_percent': 0.0},
                       {'case': 's373_n32',
                        'services': 32,
                        'time_s': 12.902027418836951,
                        'ratio': 0.33853474442505505,
                        'exact_ratio': 0.33853474442505505,
                        'loss_percent': 0.0},
                       {'case': 's389_n32',
                        'services': 32,
                        'time_s': 19.789869209751487,
                        'ratio': 0.32399690372871226,
                        'exact_ratio': 0.32399690372871226,
                        'loss_percent': 0.0},
                       {'case': 's401_n32',
                        'services': 32,
                        'time_s': 21.545747220516205,
                        'ratio': 0.33980311014844333,
                        'exact_ratio': 0.33980311014844333,
                        'loss_percent': 0.0},
                       {'case': 's419_n32',
                        'services': 32,
                        'time_s': 39.327082209289074,
                        'ratio': 0.3209678398163322,
                        'exact_ratio': 0.3209678398163322,
                        'loss_percent': 0.0},
                       {'case': 's433_n32',
                        'services': 32,
                        'time_s': 8.558219760656357,
                        'ratio': 0.3465182955767683,
                        'exact_ratio': 0.3465182955767683,
                        'loss_percent': 0.0},
                       {'case': 's449_n32',
                        'services': 32,
                        'time_s': 23.547548854723573,
                        'ratio': 0.3412590958704517,
                        'exact_ratio': 0.3412590958704517,
                        'loss_percent': 0.0},
                       {'case': 's317_n64',
                        'services': 64,
                        'time_s': 68.520506975241,
                        'ratio': 0.34139159767288935,
                        'exact_ratio': 0.34139159767288935,
                        'loss_percent': 0.0},
                       {'case': 's331_n64',
                        'services': 64,
                        'time_s': 21.43351072911173,
                        'ratio': 0.32569174458955075,
                        'exact_ratio': 0.32569174458955075,
                        'loss_percent': 0.0},
                       {'case': 's347_n64',
                        'services': 64,
                        'time_s': 123.80324439145625,
                        'ratio': 0.3307670809583909,
                        'exact_ratio': 0.3307670809583909,
                        'loss_percent': 0.0},
                       {'case': 's359_n64',
                        'services': 64,
                        'time_s': 24.71511873602867,
                        'ratio': 0.33643720092746726,
                        'exact_ratio': 0.33643720092746726,
                        'loss_percent': 0.0},
                       {'case': 's373_n64',
                        'services': 64,
                        'time_s': 41.897943675518036,
                        'ratio': 0.3336222969845459,
                        'exact_ratio': 0.3336222969845459,
                        'loss_percent': 0.0},
                       {'case': 's389_n64',
                        'services': 64,
                        'time_s': 36.861498357728124,
                        'ratio': 0.3218470342002856,
                        'exact_ratio': 0.3218470342002856,
                        'loss_percent': 0.0},
                       {'case': 's401_n64',
                        'services': 64,
                        'time_s': 62.639696944504976,
                        'ratio': 0.3317926906408422,
                        'exact_ratio': 0.3317926906408422,
                        'loss_percent': 0.0},
                       {'case': 's419_n64',
                        'services': 64,
                        'time_s': 218.58135225251317,
                        'ratio': 0.3224064766737971,
                        'exact_ratio': 0.3224064766737971,
                        'loss_percent': 0.0},
                       {'case': 's433_n64',
                        'services': 64,
                        'time_s': 77.1417781393975,
                        'ratio': 0.33516290487518474,
                        'exact_ratio': 0.33516290487518474,
                        'loss_percent': 0.0},
                       {'case': 's449_n64',
                        'services': 64,
                        'time_s': 126.21369886770844,
                        'ratio': 0.3431540730957014,
                        'exact_ratio': 0.3431540730957014,
                        'loss_percent': 0.0},
                       {'case': 's317_n128',
                        'services': 128,
                        'time_s': 1992.8307114299387,
                        'ratio': 0.339139069476943,
                        'exact_ratio': 0.339139069476943,
                        'loss_percent': 0.0},
                       {'case': 's331_n128',
                        'services': 128,
                        'time_s': 234.14887500181794,
                        'ratio': 0.32620019131790645,
                        'exact_ratio': 0.32620019131790645,
                        'loss_percent': 0.0},
                       {'case': 's347_n128',
                        'services': 128,
                        'time_s': 387.0560810919851,
                        'ratio': 0.3265298544837086,
                        'exact_ratio': 0.3265298544837086,
                        'loss_percent': 0.0},
                       {'case': 's359_n128',
                        'services': 128,
                        'time_s': 597.2555586248636,
                        'ratio': 0.3300698947880742,
                        'exact_ratio': 0.3300698947880742,
                        'loss_percent': 0.0},
                       {'case': 's373_n128',
                        'services': 128,
                        'time_s': 622.3583228755742,
                        'ratio': 0.3297017259849892,
                        'exact_ratio': 0.3297017259849892,
                        'loss_percent': 0.0},
                       {'case': 's389_n128',
                        'services': 128,
                        'time_s': 475.5327838566154,
                        'ratio': 0.3226167293239902,
                        'exact_ratio': 0.3226167293239902,
                        'loss_percent': 0.0},
                       {'case': 's401_n128',
                        'services': 128,
                        'time_s': 494.91508554294705,
                        'ratio': 0.3295106185839803,
                        'exact_ratio': 0.3295106185839803,
                        'loss_percent': 0.0},
                       {'case': 's419_n128',
                        'services': 128,
                        'time_s': 1311.105351967737,
                        'ratio': 0.3267674873785109,
                        'exact_ratio': 0.3267674873785109,
                        'loss_percent': 0.0},
                       {'case': 's433_n128',
                        'services': 128,
                        'time_s': 877.0056274533272,
                        'ratio': 0.3341173986064007,
                        'exact_ratio': 0.3341173986064007,
                        'loss_percent': 0.0},
                       {'case': 's449_n128',
                        'services': 128,
                        'time_s': 1435.5793473999947,
                        'ratio': 0.3371186622805257,
                        'exact_ratio': 0.3371186622805257,
                        'loss_percent': 0.0}],
 'greedy_beam': [{'case': 's317_n16',
                  'services': 16,
                  'time_s': 7.128409882713735,
                  'ratio': 0.33483347182584167,
                  'exact_ratio': 0.33483347182584167,
                  'loss_percent': 0.0},
                 {'case': 's331_n16',
                  'services': 16,
                  'time_s': 13.190343847032716,
                  'ratio': 0.33249959311933036,
                  'exact_ratio': 0.33249959311933036,
                  'loss_percent': 0.0},
                 {'case': 's347_n16',
                  'services': 16,
                  'time_s': 8.67006623379446,
                  'ratio': 0.33691022607935533,
                  'exact_ratio': 0.33691022607935533,
                  'loss_percent': 0.0},
                 {'case': 's359_n16',
                  'services': 16,
                  'time_s': 12.686362174003705,
                  'ratio': 0.31574179045192874,
                  'exact_ratio': 0.31574179045192874,
                  'loss_percent': 0.0},
                 {'case': 's373_n16',
                  'services': 16,
                  'time_s': 11.800802973586286,
                  'ratio': 0.3261417556392365,
                  'exact_ratio': 0.3261417556392365,
                  'loss_percent': 0.0},
                 {'case': 's389_n16',
                  'services': 16,
                  'time_s': 10.83854596440771,
                  'ratio': 0.33084554322959425,
                  'exact_ratio': 0.33084554322959425,
                  'loss_percent': 0.0},
                 {'case': 's401_n16',
                  'services': 16,
                  'time_s': 10.30661505011579,
                  'ratio': 0.3454327924145958,
                  'exact_ratio': 0.3454327924145958,
                  'loss_percent': 0.0},
                 {'case': 's419_n16',
                  'services': 16,
                  'time_s': 13.38456589216199,
                  'ratio': 0.30542451783728614,
                  'exact_ratio': 0.30542451783728614,
                  'loss_percent': 0.0},
                 {'case': 's433_n16',
                  'services': 16,
                  'time_s': 8.915501760279035,
                  'ratio': 0.35439729991815344,
                  'exact_ratio': 0.35439729991815344,
                  'loss_percent': 0.0},
                 {'case': 's449_n16',
                  'services': 16,
                  'time_s': 13.078786221904574,
                  'ratio': 0.3273896156191229,
                  'exact_ratio': 0.3273896156191229,
                  'loss_percent': 0.0},
                 {'case': 's317_n32',
                  'services': 32,
                  'time_s': 15.675142296691728,
                  'ratio': 0.349504635608175,
                  'exact_ratio': 0.34952949472549,
                  'loss_percent': 0.007112165837253048},
                 {'case': 's331_n32',
                  'services': 32,
                  'time_s': 11.839045042098821,
                  'ratio': 0.33044453156045095,
                  'exact_ratio': 0.3305103125018311,
                  'loss_percent': 0.019902840816743428},
                 {'case': 's347_n32',
                  'services': 32,
                  'time_s': 16.238882014652013,
                  'ratio': 0.3235184166476103,
                  'exact_ratio': 0.3235184166476103,
                  'loss_percent': 0.0},
                 {'case': 's359_n32',
                  'services': 32,
                  'time_s': 22.828791673535317,
                  'ratio': 0.3277153276169645,
                  'exact_ratio': 0.32779708700776017,
                  'loss_percent': 0.024942073629152794},
                 {'case': 's373_n32',
                  'services': 32,
                  'time_s': 22.666020807793714,
                  'ratio': 0.3383105947499114,
                  'exact_ratio': 0.33853474442505505,
                  'loss_percent': 0.06621171942760018},
                 {'case': 's389_n32',
                  'services': 32,
                  'time_s': 15.711281811529668,
                  'ratio': 0.32399690372871226,
                  'exact_ratio': 0.32399690372871226,
                  'loss_percent': 0.0},
                 {'case': 's401_n32',
                  'services': 32,
                  'time_s': 18.82492846355607,
                  'ratio': 0.33980311014844333,
                  'exact_ratio': 0.33980311014844333,
                  'loss_percent': 0.0},
                 {'case': 's419_n32',
                  'services': 32,
                  'time_s': 22.123971194443772,
                  'ratio': 0.3209260728647463,
                  'exact_ratio': 0.3209678398163322,
                  'loss_percent': 0.013012815118740306},
                 {'case': 's433_n32',
                  'services': 32,
                  'time_s': 17.45883986973302,
                  'ratio': 0.3465182955767683,
                  'exact_ratio': 0.3465182955767683,
                  'loss_percent': 0.0},
                 {'case': 's449_n32',
                  'services': 32,
                  'time_s': 16.633096825965875,
                  'ratio': 0.3412590958704517,
                  'exact_ratio': 0.3412590958704517,
                  'loss_percent': 0.0},
                 {'case': 's317_n64',
                  'services': 64,
                  'time_s': 30.694036179378923,
                  'ratio': 0.34136829711638716,
                  'exact_ratio': 0.34139159767288935,
                  'loss_percent': 0.006825169881452542},
                 {'case': 's331_n64',
                  'services': 64,
                  'time_s': 33.23193110104662,
                  'ratio': 0.32569174458955075,
                  'exact_ratio': 0.32569174458955075,
                  'loss_percent': 0.0},
                 {'case': 's347_n64',
                  'services': 64,
                  'time_s': 24.116942570642873,
                  'ratio': 0.3307132045705457,
                  'exact_ratio': 0.3307670809583909,
                  'loss_percent': 0.01628831614352011},
                 {'case': 's359_n64',
                  'services': 64,
                  'time_s': 42.86980774292846,
                  'ratio': 0.33643720092746726,
                  'exact_ratio': 0.33643720092746726,
                  'loss_percent': 0.0},
                 {'case': 's373_n64',
                  'services': 64,
                  'time_s': 26.792984559884022,
                  'ratio': 0.3334692860551973,
                  'exact_ratio': 0.3336222969845459,
                  'loss_percent': 0.04586352013385975},
                 {'case': 's389_n64',
                  'services': 64,
                  'time_s': 22.114907525181522,
                  'ratio': 0.32170940186238733,
                  'exact_ratio': 0.3218470342002856,
                  'loss_percent': 0.042763276734936184},
                 {'case': 's401_n64',
                  'services': 64,
                  'time_s': 31.376124156066968,
                  'ratio': 0.33167963885170776,
                  'exact_ratio': 0.3317926906408422,
                  'loss_percent': 0.03407301978717303},
                 {'case': 's419_n64',
                  'services': 64,
                  'time_s': 29.498589196931622,
                  'ratio': 0.32238024407280635,
                  'exact_ratio': 0.3224064766737971,
                  'loss_percent': 0.008136499384692186},
                 {'case': 's433_n64',
                  'services': 64,
                  'time_s': 38.31958203246557,
                  'ratio': 0.33516290487518474,
                  'exact_ratio': 0.33516290487518474,
                  'loss_percent': 0.0},
                 {'case': 's449_n64',
                  'services': 64,
                  'time_s': 30.98509493547342,
                  'ratio': 0.3431540730957014,
                  'exact_ratio': 0.3431540730957014,
                  'loss_percent': 0.0},
                 {'case': 's317_n128',
                  'services': 128,
                  'time_s': 55.31071142764147,
                  'ratio': 0.339139069476943,
                  'exact_ratio': 0.339139069476943,
                  'loss_percent': 0.0},
                 {'case': 's331_n128',
                  'services': 128,
                  'time_s': 46.99461024962737,
                  'ratio': 0.32620019131790645,
                  'exact_ratio': 0.32620019131790645,
                  'loss_percent': 0.0},
                 {'case': 's347_n128',
                  'services': 128,
                  'time_s': 50.23144160471409,
                  'ratio': 0.326503376762644,
                  'exact_ratio': 0.3265298544837086,
                  'loss_percent': 0.008108820893704127},
                 {'case': 's359_n128',
                  'services': 128,
                  'time_s': 51.234008165124955,
                  'ratio': 0.33001849593606664,
                  'exact_ratio': 0.3300698947880742,
                  'loss_percent': 0.015572111488859619},
                 {'case': 's373_n128',
                  'services': 128,
                  'time_s': 80.8782240707083,
                  'ratio': 0.3297017259849892,
                  'exact_ratio': 0.3297017259849892,
                  'loss_percent': 0.0},
                 {'case': 's389_n128',
                  'services': 128,
                  'time_s': 67.91812309250214,
                  'ratio': 0.32254641585417737,
                  'exact_ratio': 0.3226167293239902,
                  'loss_percent': 0.02179473766291684},
                 {'case': 's401_n128',
                  'services': 128,
                  'time_s': 56.49398393694256,
                  'ratio': 0.3295106185839803,
                  'exact_ratio': 0.3295106185839803,
                  'loss_percent': 0.0},
                 {'case': 's419_n128',
                  'services': 128,
                  'time_s': 80.953278435824,
                  'ratio': 0.3267349016210915,
                  'exact_ratio': 0.3267674873785109,
                  'loss_percent': 0.009972154108983666},
                 {'case': 's433_n128',
                  'services': 128,
                  'time_s': 68.28962999473896,
                  'ratio': 0.3341173986064007,
                  'exact_ratio': 0.3341173986064007,
                  'loss_percent': 0.0},
                 {'case': 's449_n128',
                  'services': 128,
                  'time_s': 71.69598902217614,
                  'ratio': 0.337118662279818,
                  'exact_ratio': 0.3371186622805257,
                  'loss_percent': 2.0993207172637085e-10}],
 'greedy': [{'case': 's317_n16',
             'services': 16,
             'time_s': 3.3451609071344137,
             'ratio': 0.3340544470337716,
             'exact_ratio': 0.33483347182584167,
             'loss_percent': 0.23266036929404654},
            {'case': 's331_n16',
             'services': 16,
             'time_s': 2.8391385041177273,
             'ratio': 0.3310404681826373,
             'exact_ratio': 0.33249959311933036,
             'loss_percent': 0.4388351044295469},
            {'case': 's347_n16',
             'services': 16,
             'time_s': 2.606155350804329,
             'ratio': 0.3362391247296323,
             'exact_ratio': 0.33691022607935533,
             'loss_percent': 0.19919292968120228},
            {'case': 's359_n16',
             'services': 16,
             'time_s': 2.236951105296612,
             'ratio': 0.314948626509777,
             'exact_ratio': 0.31574179045192874,
             'loss_percent': 0.25120651308667696},
            {'case': 's373_n16',
             'services': 16,
             'time_s': 2.6774395871907473,
             'ratio': 0.3258181265663342,
             'exact_ratio': 0.3261417556392365,
             'loss_percent': 0.09922957343133643},
            {'case': 's389_n16',
             'services': 16,
             'time_s': 3.0125524438917637,
             'ratio': 0.33030655069405407,
             'exact_ratio': 0.33084554322959425,
             'loss_percent': 0.16291364552737386},
            {'case': 's401_n16',
             'services': 16,
             'time_s': 2.5083598978817463,
             'ratio': 0.34476286263276196,
             'exact_ratio': 0.3454327924145958,
             'loss_percent': 0.19393925433395243},
            {'case': 's419_n16',
             'services': 16,
             'time_s': 2.6285918839275837,
             'ratio': 0.30473007197787044,
             'exact_ratio': 0.30542451783728614,
             'loss_percent': 0.22737069844067648},
            {'case': 's433_n16',
             'services': 16,
             'time_s': 3.41754088178277,
             'ratio': 0.35439729991815344,
             'exact_ratio': 0.35439729991815344,
             'loss_percent': 0.0},
            {'case': 's449_n16',
             'services': 16,
             'time_s': 3.0344735123217106,
             'ratio': 0.3273896156191229,
             'exact_ratio': 0.3273896156191229,
             'loss_percent': 0.0},
            {'case': 's317_n32',
             'services': 32,
             'time_s': 3.3654254619032145,
             'ratio': 0.3491662863439874,
             'exact_ratio': 0.34952949472549,
             'loss_percent': 0.10391351430524276},
            {'case': 's331_n32',
             'services': 32,
             'time_s': 3.000589909031987,
             'ratio': 0.3294181927088371,
             'exact_ratio': 0.3305103125018311,
             'loss_percent': 0.33043440754604303},
            {'case': 's347_n32',
             'services': 32,
             'time_s': 2.683197684586048,
             'ratio': 0.3228720413649684,
             'exact_ratio': 0.3235184166476103,
             'loss_percent': 0.19979551375770566},
            {'case': 's359_n32',
             'services': 32,
             'time_s': 3.4388924539089203,
             'ratio': 0.3263116740569817,
             'exact_ratio': 0.32779708700776017,
             'loss_percent': 0.453150137585967},
            {'case': 's373_n32',
             'services': 32,
             'time_s': 3.445760302245617,
             'ratio': 0.3371177570214672,
             'exact_ratio': 0.33853474442505505,
             'loss_percent': 0.4185648377079798},
            {'case': 's389_n32',
             'services': 32,
             'time_s': 3.43348790332675,
             'ratio': 0.32204347626036617,
             'exact_ratio': 0.32399690372871226,
             'loss_percent': 0.6029154741496345},
            {'case': 's401_n32',
             'services': 32,
             'time_s': 3.4519644416868687,
             'ratio': 0.339697743501489,
             'exact_ratio': 0.33980311014844333,
             'loss_percent': 0.031008146720112872},
            {'case': 's419_n32',
             'services': 32,
             'time_s': 3.525602450594306,
             'ratio': 0.3197488282349253,
             'exact_ratio': 0.3209678398163322,
             'loss_percent': 0.3797924371813921},
            {'case': 's433_n32',
             'services': 32,
             'time_s': 3.4924377985298634,
             'ratio': 0.3465182955767683,
             'exact_ratio': 0.3465182955767683,
             'loss_percent': 0.0},
            {'case': 's449_n32',
             'services': 32,
             'time_s': 3.4380823243409395,
             'ratio': 0.34073499872700547,
             'exact_ratio': 0.3412590958704517,
             'loss_percent': 0.15357748695590256},
            {'case': 's317_n64',
             'services': 64,
             'time_s': 3.608112597838044,
             'ratio': 0.34003716243741694,
             'exact_ratio': 0.34139159767288935,
             'loss_percent': 0.396739475928809},
            {'case': 's331_n64',
             'services': 64,
             'time_s': 3.2565253116190434,
             'ratio': 0.32485525652194064,
             'exact_ratio': 0.32569174458955075,
             'loss_percent': 0.2568342862556361},
            {'case': 's347_n64',
             'services': 64,
             'time_s': 3.6591129656881094,
             'ratio': 0.32957224231602983,
             'exact_ratio': 0.3307670809583909,
             'loss_percent': 0.3612326350309947},
            {'case': 's359_n64',
             'services': 64,
             'time_s': 3.673550345003605,
             'ratio': 0.334590077498287,
             'exact_ratio': 0.33643720092746726,
             'loss_percent': 0.5490247285639738},
            {'case': 's373_n64',
             'services': 64,
             'time_s': 3.7137353401631117,
             'ratio': 0.332773858616712,
             'exact_ratio': 0.3336222969845459,
             'loss_percent': 0.254311050401157},
            {'case': 's389_n64',
             'services': 64,
             'time_s': 3.7914260998368263,
             'ratio': 0.32077238974706684,
             'exact_ratio': 0.3218470342002856,
             'loss_percent': 0.33389913189320053},
            {'case': 's401_n64',
             'services': 64,
             'time_s': 3.869829883798957,
             'ratio': 0.33105342334948085,
             'exact_ratio': 0.3317926906408422,
             'loss_percent': 0.22280999919965794},
            {'case': 's419_n64',
             'services': 64,
             'time_s': 3.827966755256057,
             'ratio': 0.32111700171670216,
             'exact_ratio': 0.3224064766737971,
             'loss_percent': 0.3999531803449585},
            {'case': 's433_n64',
             'services': 64,
             'time_s': 3.6681054197251797,
             'ratio': 0.33439633709737504,
             'exact_ratio': 0.33516290487518474,
             'loss_percent': 0.22871498207570617},
            {'case': 's449_n64',
             'services': 64,
             'time_s': 3.721074853092432,
             'ratio': 0.3419973436944384,
             'exact_ratio': 0.3431540730957014,
             'loss_percent': 0.33708747526374205},
            {'case': 's317_n128',
             'services': 128,
             'time_s': 5.0992873050272465,
             'ratio': 0.3379411996338736,
             'exact_ratio': 0.339139069476943,
             'loss_percent': 0.3532090374951169},
            {'case': 's331_n128',
             'services': 128,
             'time_s': 5.197380689904094,
             'ratio': 0.3248533718712692,
             'exact_ratio': 0.32620019131790645,
             'loss_percent': 0.41288125589253877},
            {'case': 's347_n128',
             'services': 128,
             'time_s': 5.137226669117808,
             'ratio': 0.32512812604607516,
             'exact_ratio': 0.3265298544837086,
             'loss_percent': 0.42928033023190704},
            {'case': 's359_n128',
             'services': 128,
             'time_s': 5.143053136765957,
             'ratio': 0.3288257032901333,
             'exact_ratio': 0.3300698947880742,
             'loss_percent': 0.37694788818585234},
            {'case': 's373_n128',
             'services': 128,
             'time_s': 5.119546979665756,
             'ratio': 0.328608398041451,
             'exact_ratio': 0.3297017259849892,
             'loss_percent': 0.3316112283828332},
            {'case': 's389_n128',
             'services': 128,
             'time_s': 5.182669837027788,
             'ratio': 0.32172245096244123,
             'exact_ratio': 0.3226167293239902,
             'loss_percent': 0.2771952847649417},
            {'case': 's401_n128',
             'services': 128,
             'time_s': 5.265348348766565,
             'ratio': 0.32834764522517706,
             'exact_ratio': 0.3295106185839803,
             'loss_percent': 0.35293956953524885},
            {'case': 's419_n128',
             'services': 128,
             'time_s': 5.2539611253887415,
             'ratio': 0.3248622352858456,
             'exact_ratio': 0.3267674873785109,
             'loss_percent': 0.5830604837556375},
            {'case': 's433_n128',
             'services': 128,
             'time_s': 5.26818061247468,
             'ratio': 0.3331254119383714,
             'exact_ratio': 0.3341173986064007,
             'loss_percent': 0.29689763902355226},
            {'case': 's449_n128',
             'services': 128,
             'time_s': 5.329981556162238,
             'ratio': 0.33526555935442504,
             'exact_ratio': 0.3371186622805257,
             'loss_percent': 0.5496886210822338}]}


def linear_quantile(values, fraction):
    """Linear interpolation at index (N - 1) * fraction (the usual default)."""
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = int(math.floor(index))
    right = int(math.ceil(index))
    return values[left] + (values[right] - values[left]) * (index - left)


def validate_data():
    encoded = json.dumps(DATA, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != DATA_SHA256:
        raise ValueError("The embedded scaling observations have changed")
    if set(DATA) != set(METHODS):
        raise ValueError("Expected exactly the four declared methods")
    for method, rows in DATA.items():
        if len(rows) != 40 or {row["case"] for row in rows} != EXPECTED:
            raise ValueError(f"Incorrect workload set for {method}")
        for row in rows:
            if row["services"] != int(row["case"].split("_n")[1]):
                raise ValueError(f"Inconsistent size: {row}")
            for key in ("time_s", "ratio", "exact_ratio"):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f"Invalid {key}: {row}")
            if row.get("status") == "timeout_censored":
                raise ValueError("Censored runs require a different rendering")
            expected_loss = 100 * (1 - row["ratio"] / row["exact_ratio"])
            if not math.isclose(row["loss_percent"], expected_loss,
                                rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"Incorrect paired loss: {row}")
            if row["ratio"] > row["exact_ratio"] + 1e-7:
                raise ValueError(f"Value exceeds its exact reference: {row}")
        for size in SIZES:
            if sum(row["services"] == size for row in rows) != 10:
                raise ValueError(f"Expected ten cases at size {size}")
    for case in EXPECTED:
        references = [next(row["exact_ratio"] for row in DATA[method]
                           if row["case"] == case) for method in METHODS]
        if not all(math.isclose(value, references[0], abs_tol=1e-12)
                   for value in references):
            raise ValueError(f"Methods do not share the paired reference: {case}")


def aggregate():
    summary = {}
    for method in METHODS:
        summary[method] = []
        for size in SIZES:
            rows = [row for row in DATA[method] if row["services"] == size]
            times = [row["time_s"] for row in rows]
            summary[method].append({
                "services": size, "cases": len(rows),
                "mean_time_s": mean(times),
                "q25_time_s": linear_quantile(times, .25),
                "q75_time_s": linear_quantile(times, .75),
                "min_time_s": min(times), "max_time_s": max(times),
                "mean_loss_percent": mean(row["loss_percent"] for row in rows),
            })
    return summary


def place_loss_labels(fig, ax, summary, marker_positions):
    """Choose nearby clear labels; keep Beam/16 in the clear space above it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes, records, labels = [], [], []
    points = [(ax.transData.transform((x, y)), radius * fig.dpi / 72)
              for x, y, radius in marker_positions]
    curve_paths = [line.get_transform().transform_path(line.get_path())
                   for line in ax.lines]
    # Search by distance, not by rows of vertical offsets: a nearby higher
    # position is preferred over a long horizontal detour. Small steps expose
    # the narrow clear spaces between observations in the first two groups.
    for method in ("greedy", "greedy_beam"):
        color = STYLES[method][1]
        for item in summary[method]:
            size = item["services"]
            value = item["mean_time_s"]
            vertical_offsets = list(range(3,44,2))
            if method == "greedy_beam" and size == 32:
                vertical_offsets += list(range(-17,-4,2))
            candidates = sorted(((dx,dy) for dx in range(-36,37,2)
                                 for dy in vertical_offsets),
                                key=lambda p:(math.hypot(p[0],p[1]+3)+(2 if p[1]<0 else 0),abs(p[0]),abs(p[1]),p[0]))
            chosen = None
            text = ax.annotate(
                f'{item["mean_loss_percent"]:.2f}%', (size, value),
                xytext=(0, 3), textcoords="offset points", ha="center",
                va="bottom", fontsize=6.15, color=color, zorder=9,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": .95, "pad": .25})
            for dx, dy in candidates:
                text.set_position((dx,dy))
                text.update_positions(renderer)
                text.update_bbox_position_size(renderer)
                box = text.get_bbox_patch().get_window_extent(renderer)
                padded = box.padded(.65 * fig.dpi / 72)
                in_axes = (padded.x0 > ax.bbox.x0 + 1 and
                           padded.x1 < ax.bbox.x1 - 1 and
                           padded.y0 > ax.bbox.y0 + 1 and
                           padded.y1 < ax.bbox.y1 - 1)
                hits_point = any(
                    math.hypot(max(box.x0 - xy[0], 0, xy[0] - box.x1),
                               max(box.y0 - xy[1], 0, xy[1] - box.y1)) < radius + .5*fig.dpi/72
                    for xy, radius in points)
                hits_curve = any(path.intersects_bbox(padded,filled=False)
                                 for path in curve_paths)
                if in_axes and not hits_point and not hits_curve and not any(
                        padded.overlaps(old) for old in boxes):
                    chosen = (text, padded, dx, dy)
                    break
            if chosen is None:
                raise RuntimeError(f"Could not safely place loss label {method}/{size}")
            text, box, dx, dy = chosen
            boxes.append(box)
            labels.append(text)
            records.append({"method": method, "services": size,
                            "text": text.get_text(),
                            "mean_loss_percent": item["mean_loss_percent"],
                            "offset_points": [dx, dy]})
    return labels, records


def render(outdir, dpi):
    validate_data()
    summary = aggregate()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "text.usetex": False,
        "text.color": "#39454f", "axes.labelcolor": "#39454f",
        "xtick.color": "#39454f", "ytick.color": "#39454f",
        "axes.edgecolor": "#7d898e",
        "figure.autolayout": False, "figure.constrained_layout.use": False,
        "font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    })
    width, height = 3.45, 3.05
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([.175, .35325 / height, .795, 2.3115 / height])
    marker_positions = []
    handles = []
    all_times = [row["time_s"] for rows in DATA.values() for row in rows]
    for index, method in enumerate(METHODS):
        label, color, marker, linestyle = STYLES[method]
        items = summary[method]
        times = [item["mean_time_s"] for item in items]
        ax.fill_between(SIZES, [item["q25_time_s"] for item in items],
                        [item["q75_time_s"] for item in items],
                        color=color, alpha=.10, linewidth=0, zorder=1)
        for size in SIZES:
            rows = sorted((row for row in DATA[method] if row["services"] == size),
                          key=lambda row: int(row["case"].split("_")[0][1:]))
            # Fixed, deterministic horizontal jitter only; y values are the
            # explicit planning-time values in DATA.
            offsets = [-2.7 + .6 * i for i in range(10)]
            xs = [size + offset for offset in offsets]
            ys = [row["time_s"] for row in rows]
            ax.scatter(xs, ys, s=5.8, color=color, alpha=.52,
                       linewidths=0, zorder=2)
            marker_positions.extend((x, y, 1.25) for x, y in zip(xs, ys))
        ax.plot(SIZES, times, color=color, linestyle=linestyle,
                linewidth=1.05, marker=marker, markersize=4.0,
                markerfacecolor=color, markeredgewidth=.85,
                zorder=4 + index * .1)
        marker_positions.extend((size, value, 2.2)
                                for size, value in zip(SIZES, times))
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle,
                              linewidth=1.05, marker=marker, markersize=3.8,
                              markerfacecolor=color, markeredgewidth=.8,
                              label=label))
    lower = min(1.8, min(all_times) / 1.5)
    upper = max(3000, max(all_times) * 1.35)
    ax.set(xlabel="Number of services", ylabel="Total planning time (s)",
           xlim=(8, 136), ylim=(lower, upper), yscale="log")
    ax.set_xticks(SIZES)
    ticks = [10. ** exponent * multiplier
             for exponent in range(-5, 7) for multiplier in (1, 3)]
    ticks = [value for value in ticks if lower <= value <= upper]
    ax.set_yticks(ticks, [f"{value:g}" for value in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", linewidth=.4, color="#d9d9d9")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=2, length=2.5)
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 4
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.53, .994), fontsize=7.0, handlelength=1.7,
               handletextpad=.45, columnspacing=1.2, labelspacing=.35,
               borderaxespad=.1)
    labels, label_records = place_loss_labels(fig, ax, summary, marker_positions)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if (box.x0 < -.5 or box.y0 < -.5 or
                box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5):
            outside.append(text.get_text())
    if outside:
        raise RuntimeError(f"Text outside the page: {outside}")
    # Crop horizontal whitespace while keeping the full vertical layout.
    tight = fig.get_tightbbox(renderer)
    padding = .02
    export_box = Bbox.from_extents(tight.x0 - padding, 0,
                                   tight.x1 + padding, height)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "scalability"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                bbox_inches=export_box, pad_inches=0, metadata={
        "Title": "Planner scaling on ten workloads per size",
        "Creator": Path(__file__).name, "CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white",
                bbox_inches=export_box, pad_inches=0)
    qa = {"source_data_sha256": SOURCE_SHA256,
          "font_family": "DejaVu Sans", "matplotlib_version": matplotlib.__version__,
          "observations": 160, "cases": 40, "cases_per_size": 10,
          "page_width_inches": export_box.width, "page_height_inches": height,
          "time_aggregation": "arithmetic mean of the planning-time values",
          "dispersion": "all points and interquartile range (linear interpolation)",
          "exact_loss_labels": False,
          "loss_aggregation": "mean of paired losses against the same input optimum",
          "text_outside_page": outside, "loss_labels": label_records,
          "layout_checks": "Nearby clear positions with Beam/16 above its point; labels cover no observations or curves and do not overlap each other",
          "footer_notes": False,
          "aggregates": summary}
    stem.with_suffix(".qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    caption = (
        "Planning time and solution quality for 40 workloads (ten per size). "
        "Lines show arithmetic means, shaded bands the interquartile range, "
        "and small points all individual workloads; the vertical axis is logarithmic. "
        "Percentages give the mean paired utility-to-power ratio loss relative "
        "to the exact reference for the same input, computed as 100(1 - R/R*). "
        "The ten workloads per size are "
        "different inputs, not temporal repetitions or evidence of asymptotic complexity.")
    stem.with_suffix(".caption.txt").write_text(caption + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"Created {stem}.pdf and .png; 160 literal observations, layout checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("scalability_figure"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        render(args.outdir, args.dpi)


if __name__ == "__main__":
    main()
