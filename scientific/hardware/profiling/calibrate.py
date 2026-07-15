#!/usr/bin/env python3
"""Calibrate configurations before freezing the deployment comparison.

The worker and telemetry code are frozen copies from the separate batch study.
Only this entry point controls the new protocol. No original records are edited.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import statistics
import time
import traceback

import batch_worker as worker
import gpu_support as gpu
from cleanup import destroy_layout_retry


def idle(nv, path, seconds=12):
    begin = time.monotonic() + .1
    samples = [nv.sample()]
    while time.monotonic() < begin + seconds:
        time.sleep(.02)
        samples.append(nv.sample())
    result = dict(begin=begin, duration=seconds, samples=samples)
    gpu.atomic_json(path, result)
    return result


def run(args):
    # Match the profiling and replay locks before touching the same device.
    import fcntl
    if os.geteuid() == 0:
        raise RuntimeError('Run Python as an ordinary user; only nvidia-smi is elevated')
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(f'/tmp/efficient_mig_gpu{args.gpu}.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu.assert_empty(args.gpu)
    initial = gpu.smi('-i', args.gpu, '-q')
    if 'Enabled' not in initial.split('MIG Mode', 1)[1].split('Accounting', 1)[0]:
        raise RuntimeError('MIG must already be enabled')
    (root/'initial_gpu.txt').write_text(initial)
    cap = float(gpu.smi('-i', args.gpu, '--query-gpu=power.limit', '--format=csv,noheader,nounits'))
    if abs(cap-165) > .01:
        raise RuntimeError('Calibration requires the recorded 165 W cap')
    plan = json.loads(Path(args.plan).read_text())
    configs = [c for c in plan['configurations'] if c['lane'] == args.lane]
    if args.only:
        configs = [c for c in configs if c['id'] == args.only]
    if not configs:
        raise ValueError('No configurations for this lane')
    for config in configs:
        gpu.safe_name(config['id'])
    uuid = gpu.smi('-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader')
    gpu.atomic_json(root/'launches'/f'{time.time_ns()}.json', dict(
        host=socket.gethostname(), gpu=args.gpu, gpu_uuid=uuid,
        lane=args.lane, pid=os.getpid(), cap_W=cap,
        plan_sha256=gpu.sha_file(args.plan),
        source_sha256={p.name:gpu.sha_file(p) for p in Path(__file__).parent.glob('*.py')},
        configurations=[c['id'] for c in configs]))
    nv = gpu.NVML(args.gpu)
    owned = actor = None
    outcomes = []
    try:
        for config in configs:
            base = root/'raw'/config['id']
            if (base/'COMPLETE').exists():
                record = json.loads((base/'COMPLETE').read_text())
                if record['config'] != config:
                    raise RuntimeError('Completed-run configuration mismatch')
                for name, digest in record['sha256'].items():
                    if gpu.sha_file(base/name) != digest:
                        raise RuntimeError('Completed-run integrity mismatch: ' + name)
                outcomes.append(dict(id=config['id'],status='previously_complete'))
                continue
            if base.exists():
                raise RuntimeError(f'Partial record requires explicit recovery: {base}')
            base.mkdir(parents=True)
            gpu.atomic_json(base/'config.json', dict(config, gpu_uuid=uuid, host=socket.gethostname()))
            success = False
            try:
                gpu.assert_empty(args.gpu)
                gpu.smi('-i', args.gpu, '-lgc', f"{config['frequency_MHz']},{config['frequency_MHz']}")
                time.sleep(3)
                idle(nv, base/'baseline_pre.json')
                assigned, owned = gpu.create_layout(args.gpu, [config['slices']], base/'ownership.json')
                actor = worker.start_actor(assigned[0]['uuid'], config, args)
                gpu.atomic_json(base/'actor.json', actor[3])
                time.sleep(3)
                worker.measure(nv, actor, base, 'resident_pre', 'idle', 0, 6, 0)
                busy = worker.measure(nv, actor, base, 'busy', 'saturation', 0, 15, 0)
                durations = [x['wall_L'] for x in busy['results'][0]['records']]
                mu = config['batch']/statistics.mean(durations)
                rates = sorted(set([min(1., .1*mu), .1*mu, .3*mu, .6*mu, .85*mu]))
                gpu.atomic_json(base/'rates.json', dict(mu=mu, rates=rates))
                for index, rate in enumerate(rates):
                    wanted = max(15., 20*config['batch']/rate)
                    duration = min(120., wanted)
                    seed = int.from_bytes(hashlib.sha256(
                        f"deployment-calibration|{config['id']}|{index}".encode()).digest()[:8], 'big')
                    worker.measure(nv, actor, base, f'load_{index}', 'poisson', rate,
                        duration, seed, dict(expected_batches=duration*rate/config['batch'],
                                             duration_capped=wanted>120, mu=mu))
                worker.measure(nv, actor, base, 'resident_post', 'idle', 0, 6, 0)
                worker.stop_actor(actor)
                actor = None
                cleanup = []
                destroy_layout_retry(args.gpu, owned, cleanup)
                gpu.atomic_json(base/'cleanup.json', cleanup)
                owned = None
                time.sleep(3)
                idle(nv, base/'baseline_post.json')
                success = True
            except BaseException as exc:
                gpu.atomic_json(base/'error.json', dict(error=traceback.format_exc(), type=type(exc).__name__))
                outcomes.append(dict(id=config['id'],status='error',error=str(exc)))
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                if actor is not None:
                    worker.stop_actor(actor)
                    actor = None
                if owned is not None:
                    cleanup = []
                    try:
                        destroy_layout_retry(args.gpu, owned, cleanup)
                    finally:
                        gpu.atomic_json(base/'cleanup_recovery.json', cleanup)
                    owned = None
                gpu.unlock_empty_gpu(args.gpu)
            if success:
                gpu.atomic_json(base/'COMPLETE', dict(config=config, sha256={
                    p.name:gpu.sha_file(p) for p in sorted(base.glob('*.json'))}))
                outcomes.append(dict(id=config['id'],status='complete'))
            gpu.atomic_json(root/'progress.json', dict(outcomes=outcomes))
            print(json.dumps(outcomes[-1]), flush=True)
        gpu.atomic_json(root/'finished.json', dict(outcomes=outcomes))
    finally:
        if actor is not None:
            worker.stop_actor(actor)
        if owned is not None:
            destroy_layout_retry(args.gpu, owned)
        try:
            gpu.unlock_empty_gpu(args.gpu)
        finally:
            nv.close()
        gpu.atomic_json(root/'restoration.json', dict(empty=gpu.assert_empty(args.gpu),
            clocks='unlocked', cap_W=cap, persistence='unchanged', mig='enabled'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--lane', type=int, required=True)
    parser.add_argument('--plan', default='plan.json')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--torch-home', required=True)
    parser.add_argument('--only')
    args = parser.parse_args()
    mp.set_start_method('spawn')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    run(args)
