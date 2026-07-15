#!/usr/bin/env python3
"""Replay arbitrary deployment plans on independent logical A30 boards.

Each service has one immutable Poisson trace shared by every policy. Requests
are routed to replicas using seeded categorical draws, then processed in FIFO
batches. A final incomplete batch is padded at the end of the arrival window;
padding is recorded and never counted as offered work or utility.

Only --run uses CUDA or changes a GPU. --prepare and --dry-run are CPU-only.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import re
import signal
import socket
import statistics
import sys
import time
import traceback
from urllib.parse import urlparse

import gpu_support as gpu

SCHEMA = 'deployment_replay_20261022_v1'
WEIGHTS = {
    'fasterrcnn_resnet50_fpn_v2': 'FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1',
    'deeplabv3_resnet50': 'DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1',
    'lraspp_mobilenet_v3_large': 'LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1',
    'fcn_resnet50': 'FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1',
    'retinanet_resnet50_fpn': 'RetinaNet_ResNet50_FPN_Weights.COCO_V1',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def safe_name(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError('Identifiers must be safe path components: ' + value)
    return value


def positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + ' must be positive and finite')
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + ' must be positive and finite')
    return number


def integer(value, name, minimum=1):
    """Accept integral JSON values without silently changing a deployment."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be an integer')
    if not math.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError(name + f' must be an integer >= {minimum}')
    return int(value)


def normalize_selection(document):
    """Validate explicit plans, expand multiplicities and check shared demand."""
    cases = []
    seen_cases = set()
    for original in document['cases']:
        cid = safe_name(original['case_id'])
        if cid in seen_cases:
            raise ValueError('Duplicate case: ' + cid)
        seen_cases.add(cid)
        services = {}
        raw_services = original['services']
        if isinstance(raw_services, dict):
            raw_services = [dict(v, service_id=k) for k, v in raw_services.items()]
        for service_index, row in enumerate(raw_services):
            sid = str(row.get('service_id', service_index))
            if sid in services:
                raise ValueError('Duplicate service: ' + sid)
            services[sid] = dict(service_id=sid,
                                lambda_rps=positive(row.get('lambda_rps', row.get('lam')), 'lambda_rps'),
                                slo_s=positive(row.get('slo_s', row.get('slo')), 'slo_s'))
        policies = {}
        for name, policy in original['policies'].items():
            name = safe_name(name)
            if policy is None or policy.get('feasible') is False:
                continue
            boards = []
            ids, replica_ids = set(), set()
            rates = defaultdict(float)
            variants = defaultdict(set)
            for source in policy['gpus']:
                gid = integer(source['logical_gpu'], 'logical_gpu', minimum=0)
                if gid in ids or gid < 0:
                    raise ValueError('Invalid or duplicate logical GPU')
                ids.add(gid)
                mode = source.get('mode', 'active' if source['replicas'] else 'idle_fixed')
                if mode not in ('active', 'idle_fixed', 'idle_unlocked'):
                    raise ValueError('Unknown GPU mode')
                frequency = integer(source.get('frequency_MHz', 0), 'frequency_MHz', minimum=0)
                cap = positive(source.get('power_cap_W', original.get('cap_W')), 'power_cap_W')
                if frequency <= 0 and mode != 'idle_unlocked':
                    raise ValueError('Invalid frequency')
                replicas = []
                for entry_index, entry in enumerate(source['replicas']):
                    sid = str(entry.get('service_id', entry.get('service')))
                    if sid not in services:
                        raise ValueError('Unknown service ' + sid)
                    arch = entry['arch']
                    batch = integer(entry.get('B', entry.get('batch', entry.get('batch_size', 1))), 'batch')
                    slices = integer(entry['slices'], 'slices')
                    rate = positive(entry.get('rate_rps', entry.get('rate_requests_s')), 'rate_rps')
                    count = integer(entry.get('n', 1), 'n')
                    if arch not in WEIGHTS or slices not in gpu.MIG or batch < 1 or count < 1:
                        raise ValueError('Unsupported configuration')
                    if entry.get('frequency_MHz', frequency) != frequency:
                        raise ValueError('A replica frequency differs from its board frequency')
                    utility = float(entry.get('utility_per_request', 1.0))
                    if not math.isfinite(utility) or utility < 0:
                        raise ValueError('Invalid utility')
                    precision = str(entry.get('precision', 'FP32')).upper()
                    variants[sid].add((arch, batch, precision))
                    if precision != 'FP32':
                        raise ValueError('Only FP32 inference is implemented; no silent precision conversion')
                    for copy in range(count):
                        rid = str(entry.get('replica_id', f'g{gid}_s{sid}_a{entry_index}'))
                        if count > 1:
                            rid += f'_copy{copy}'
                        if rid in replica_ids:
                            raise ValueError('Replica IDs must be distinct across the whole deployment')
                        replica_ids.add(rid)
                        replica = dict(replica_id=rid, service_id=sid, arch=arch, B=batch,
                                       slices=slices, rate_rps=rate, utility_per_request=utility,
                                       slo_s=services[sid]['slo_s'],
                                       catalog_id=entry.get('catalog_id'),
                                       physical_config_id=entry.get('physical_config_id'),
                                       input_spec=entry.get('input_spec'),
                                       caller_shape=entry.get('caller_shape', (entry.get('input_spec') or {}).get('caller_shape', [3, 640, 640] if
                                           arch.startswith(('faster', 'retina')) else [3, 520, 520])))
                        replicas.append(replica)
                        rates[sid] += rate
                layout_spec = source.get('layout_slices')
                if layout_spec is None:
                    layout_spec = source.get('provisioned_mig_layout')
                if layout_spec is None:
                    layout_spec = [r['slices'] for r in replicas]
                layout = [integer(x, 'layout slice') for x in layout_spec]
                if sum(layout) > 4 or any(x not in gpu.MIG for x in layout):
                    raise ValueError('Invalid A30 layout')
                remaining = list(layout)
                for replica in replicas:
                    if replica['slices'] not in remaining:
                        raise ValueError('Replicas do not fit their declared layout')
                    remaining.remove(replica['slices'])
                if mode == 'idle_unlocked' and (replicas or layout):
                    raise ValueError('An unlocked idle GPU cannot contain replicas or MIG instances')
                if mode == 'idle_fixed' and replicas:
                    raise ValueError('An idle GPU cannot contain active replicas')
                boards.append(dict(logical_gpu=gid, frequency_MHz=frequency, mode=mode,
                                   power_cap_W=cap, layout_slices=layout, replicas=replicas))
            for sid, service in services.items():
                if not math.isclose(rates[sid], service['lambda_rps'], rel_tol=1e-7, abs_tol=1e-7):
                    raise ValueError(f'{cid}/{name}/{sid}: replica rates do not sum to service demand')
                if len(variants[sid]) != 1:
                    raise ValueError('A service must use one variant, including batch size')
            policies[name] = dict(gpus=sorted(boards, key=lambda b: b['logical_gpu']),
                                  predicted=policy.get('predicted', {}))
        if not policies:
            continue
        # All explicit fleet boards count, including boards left empty by a policy.
        fleets = [{b['logical_gpu'] for b in p['gpus']} for p in policies.values()]
        if any(f != fleets[0] for f in fleets):
            raise ValueError('Every policy must list the same logical fleet, including empty GPUs')
        cases.append(dict(case_id=cid, trace_id=str(original.get('trace_id', cid)),
                          services=services, policies=policies))
    if not cases:
        raise ValueError('No feasible selected deployment')
    return cases


def service_trace(case, service, repetition, warm_s, duration_s):
    """Arrival times and image identities do not depend on the scheduler."""
    arrival_seed = seed(SCHEMA, case['trace_id'], service['service_id'], repetition, 'arrivals')
    rng = random.Random(arrival_seed)
    elapsed, requests = 0.0, []
    while True:
        elapsed += rng.expovariate(service['lambda_rps'])
        if elapsed >= warm_s + duration_s:
            break
        request_id = len(requests)
        requests.append(dict(request_id=request_id, arrival_offset=elapsed,
                             image_id=request_id % 8, measured=elapsed >= warm_s))
    return dict(service_id=service['service_id'], lambda_rps=service['lambda_rps'],
                seed=arrival_seed, window_start_offset=warm_s,
                window_end_offset=warm_s + duration_s, requests=requests,
                trace_sha256=digest(requests))


def route_trace(trace, replicas, trace_id, repetition):
    """Seeded categorical thinning preserves the Poisson per-replica model."""
    replicas = sorted(replicas, key=lambda r: r['replica_id'])
    cumulative, total = [], 0.0
    for replica in replicas:
        total += replica['rate_rps']
        cumulative.append(total)
    if not math.isclose(total, trace['lambda_rps'], rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError('Routing weights must sum to the common trace rate')
    rng = random.Random(seed(SCHEMA, trace_id, trace['service_id'], repetition, 'routing'))
    routed = {r['replica_id']: [] for r in replicas}
    for request in trace['requests']:
        index = min(bisect_right(cumulative, rng.random() * total), len(replicas) - 1)
        routed[replicas[index]['replica_id']].append(request)
    flat = [row['request_id'] for rows in routed.values() for row in rows]
    if sorted(flat) != [r['request_id'] for r in trace['requests']]:
        raise AssertionError('Routing lost or duplicated requests')
    return routed


def batch_groups(requests, batch, window_end):
    """Only the terminal batch may be padded, and only once the trace ends."""
    for begin in range(0, len(requests), batch):
        rows = requests[begin:begin + batch]
        padding = batch - len(rows)
        ready = window_end if padding else rows[-1]['arrival_offset']
        yield dict(requests=rows, padding=padding, ready_offset=ready,
                   image_ids=[r['image_id'] for r in rows] + [rows[-1]['image_id']] * padding)


def prepare_jobs(cases, repetitions, warm_s, duration_s):
    jobs, traces = [], {}
    for repetition in range(repetitions):
        for case in cases:
            cid = case['case_id']
            service_traces = {sid: service_trace(case, service, repetition, warm_s, duration_s)
                              for sid, service in case['services'].items()}
            traces[f'{cid}__r{repetition}'] = service_traces
            policies = sorted(case['policies'])
            rotate = repetition % len(policies)
            policies = policies[rotate:] + policies[:rotate]
            for name in policies:
                policy = case['policies'][name]
                all_replicas = [r for board in policy['gpus'] for r in board['replicas']]
                routes = {}
                for sid, trace in service_traces.items():
                    routes.update(route_trace(trace, [r for r in all_replicas if r['service_id'] == sid],
                                              case['trace_id'], repetition))
                for board in policy['gpus']:
                    job = dict(case_id=cid, trace_id=case['trace_id'], policy=name,
                               repetition=repetition, **board)
                    job['plans'] = [dict(replica_id=r['replica_id'],
                                         window_start_offset=warm_s, window_end_offset=warm_s + duration_s,
                                         service_trace_sha256=service_traces[r['service_id']]['trace_sha256'],
                                         requests=routes[r['replica_id']]) for r in board['replicas']]
                    jobs.append(job)
    # Assign hardware before filtering cases or policies. The rotation depends
    # only on the trace/cap cell, so different SLOs retain valid aliases and a
    # particular logical GPU is not tied to one physical device throughout.
    groups = sorted({(job['trace_id'], job['power_cap_W']) for job in jobs})
    group_index = {group: index for index, group in enumerate(groups)}
    for job in jobs:
        job['lane_group_index'] = group_index[(job['trace_id'], job['power_cap_W'])]
    return jobs, traces


def lane_for_job(job, lanes):
    if lanes < 1:
        raise ValueError('At least one physical lane is required')
    return (job['logical_gpu'] + job['repetition'] + job['lane_group_index']) % lanes


def integrate(samples, start, end):
    if end <= start:
        return 0.0
    if not samples or samples[0]['t'] > start or samples[-1]['t'] < end:
        return None
    total = 0.0
    for a, b in zip(samples, samples[1:]):
        if b['t'] <= a['t']:
            raise ValueError('Non-increasing telemetry timestamps')
        lo, hi = max(start, a['t']), min(end, b['t'])
        if hi <= lo:
            continue
        span = b['t'] - a['t']
        p0 = a['power_W'] + (b['power_W'] - a['power_W']) * (lo - a['t']) / span
        p1 = a['power_W'] + (b['power_W'] - a['power_W']) * (hi - a['t']) / span
        total += (hi - lo) * (p0 + p1) / 2
    return total


def timing_summary(records, expected, slo_s):
    rows = [r for r in records if r['measured']]
    expected_ids = {r['request_id'] for r in expected if r['measured']}
    actual_ids = [r['request_id'] for r in rows]
    if len(actual_ids) != len(set(actual_ids)) or not set(actual_ids) <= expected_ids:
        raise ValueError('Duplicate or unplanned measured request')
    values = sorted(r['response_s'] for r in rows)
    mean = statistics.mean(values) if values else None
    without_padding = [r['response_s'] for r in rows if not r['padded_batch']]
    return dict(offered=len(expected_ids), completed=len(rows),
                unfinished_request_ids=sorted(expected_ids - set(actual_ids)),
                mean_response_s=mean, slo_s=slo_s,
                mean_slo_met=(mean <= slo_s) if mean is not None and len(rows) == len(expected_ids) else None,
                p95_response_s=values[min(len(values)-1, math.ceil(.95*len(values))-1)] if values else None,
                request_slo_misses=sum(x > slo_s for x in values),
                padded_batch_measured_requests=sum(r['padded_batch'] for r in rows),
                mean_response_without_final_padding_s=statistics.mean(without_padding) if without_padding else None)


def telemetry_summary(samples, start, end, frequency, cap):
    selected = [s for s in samples if start <= s['t'] <= end]
    energy = integrate(samples, start, end)
    return dict(energy_J=energy, duration_s=end-start,
                mean_power_W=energy/(end-start) if energy is not None and end > start else None,
                sample_count=len(selected),
                max_sample_gap_s=max((b['t']-a['t'] for a, b in zip(samples, samples[1:])), default=None),
                power_cap_flag_samples=sum(bool((s.get('throttle') or 0) & 4) for s in selected),
                max_sampled_power_W=max((s['power_W'] for s in selected), default=None),
                samples_above_cap=sum(s['power_W'] > cap for s in selected),
                min_clock_MHz=min((s['clock_MHz'] for s in selected if s.get('clock_MHz') is not None), default=None),
                below_requested_clock_samples=sum(s.get('clock_MHz') is not None and s['clock_MHz'] < frequency for s in selected),
                throttle_masks=sorted({s['throttle'] for s in selected if s.get('throttle') is not None}))


def child(uuid, replica, settings, commands, answers):
    records, batches, block_id = [], [], None
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          TORCH_HOME=settings['torch_home'])
        import numpy as np
        import torch
        import torchvision
        from PIL import Image
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        if not torch.__version__.startswith('2.6.0') or not torchvision.__version__.startswith('0.21.0'):
            raise RuntimeError('The protocol requires PyTorch 2.6.0 and torchvision 0.21.0')
        arch, batch = replica['arch'], replica['B']
        detection = arch.startswith(('faster', 'retina'))
        module = torchvision.models.detection if detection else torchvision.models.segmentation
        weight_class, member = WEIGHTS[arch].split('.')
        weights = getattr(getattr(module, weight_class), member)
        checkpoint = Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(weights.url).path).name
        if not checkpoint.is_file():
            raise RuntimeError('Checkpoint must be staged before replay: ' + str(checkpoint))
        model = getattr(module, arch)(weights=weights).eval().cuda()
        files = sorted(p for p in Path(settings['images']).iterdir()
                       if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:8]
        if len(files) != 8:
            raise RuntimeError('Eight archived real images are required')
        height, width = replica['caller_shape'][-2:]
        tensors, input_hashes = [], []
        for path in files:
            array = np.asarray(Image.open(path).convert('RGB').resize((width, height))).copy()
            x = torch.from_numpy(array).permute(2, 0, 1).float().div(255)
            if not detection:
                x = (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            input_hashes.append(hashlib.sha256(x.numpy().tobytes()).hexdigest())
            tensors.append(x.cuda())

        def infer(image_ids):
            # Assembly is included in replica execution wall time, not hidden in calibration.
            start = time.monotonic()
            x = [tensors[i] for i in image_ids]
            if not detection:
                x = torch.stack(x)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            model(x)
            b.record()
            torch.cuda.synchronize()
            finish = time.monotonic()
            return start, finish, a.elapsed_time(b) / 1000

        with torch.inference_mode():
            warmup = [infer([(i*batch+j) % 8 for j in range(batch)])[1:]
                      for i in range(settings['warmup_batches'])]
            answers.put(dict(kind='ready', replica_id=replica['replica_id'],
                             torch=torch.__version__, torchvision=torchvision.__version__,
                             python=sys.version, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                             weights=str(weights), checkpoint_sha256=file_hash(checkpoint),
                             caller_shape=replica['caller_shape'], B=batch,
                             dtype=str(next(model.parameters()).dtype),
                             tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                             tf32_cudnn=torch.backends.cudnn.allow_tf32,
                             input_sha256=input_hashes,
                             images=[dict(name=p.name, sha256=file_hash(p)) for p in files],
                             detection_transform=str(model.transform) if detection else None,
                             warmup_batches=len(warmup)))
            while True:
                command = commands.get()
                if command['op'] == 'stop':
                    return
                if command['op'] != 'prepare':
                    raise RuntimeError('Expected PREPARE')
                plan, block_id = command['plan'], command['block_id']
                records, batches = [], []
                answers.put(dict(kind='prepared', block_id=block_id))
                go = commands.get()
                if go.get('op') != 'go' or go.get('block_id') != block_id:
                    raise RuntimeError('Invalid GO')
                origin = go['origin']
                deadline = origin + plan['window_end_offset'] + settings['drain_s']
                if time.monotonic() >= origin:
                    raise RuntimeError('Missed synchronized trace origin')
                sent, last_progress = 0, time.monotonic()
                for batch_index, group in enumerate(batch_groups(plan['requests'], batch, plan['window_end_offset'])):
                    ready = origin + group['ready_offset']
                    while time.monotonic() < ready:
                        time.sleep(min(.01, max(0, ready-time.monotonic())))
                    if time.monotonic() >= deadline:
                        break
                    start, finish, cuda = infer(group['image_ids'])
                    batches.append(dict(batch_index=batch_index, start=start, finish=finish,
                                        execution_s=finish-start, cuda_s=cuda, padding=group['padding'],
                                        request_ids=[r['request_id'] for r in group['requests']],
                                        image_ids=group['image_ids'], ready=ready))
                    for request in group['requests']:
                        arrival = origin + request['arrival_offset']
                        records.append(dict(request_id=request['request_id'], image_id=request['image_id'],
                                            arrival=arrival, start=start, finish=finish,
                                            response_s=finish-arrival, batch_fill_s=ready-arrival,
                                            queue_s=max(0, start-ready), execution_s=finish-start,
                                            measured=request['measured'], batch_index=batch_index,
                                            padded_batch=bool(group['padding']),
                                            completed_before_deadline=finish <= deadline))
                    if time.monotonic() - last_progress >= 1:
                        answers.put(dict(kind='progress', block_id=block_id, records=records[sent:]))
                        sent, last_progress = len(records), time.monotonic()
                completed = {r['request_id'] for r in records}
                unfinished = [r['request_id'] for r in plan['requests'] if r['request_id'] not in completed]
                answers.put(dict(kind='result', block_id=block_id, replica_id=replica['replica_id'],
                                 records=records, batches=batches, unfinished_request_ids=unfinished,
                                 drain_capped=bool(unfinished) or any(not r['completed_before_deadline'] for r in records),
                                 total_padding=sum(b['padding'] for b in batches), finished=time.monotonic()))
                block_id, records, batches = None, [], []
    except BaseException:
        answers.put(dict(kind='error', block_id=block_id, error=traceback.format_exc(),
                         partial_records=records, partial_batches=batches))


class Actor:
    def __init__(self, uuid, replica, args):
        ctx = mp.get_context('spawn')
        self.commands, self.answers = ctx.Queue(), ctx.Queue()
        settings = {key: getattr(args, key) for key in ('torch_home', 'images', 'drain_s', 'warmup_batches')}
        self.process = ctx.Process(target=child, args=(uuid, replica, settings, self.commands, self.answers))
        self.process.start()

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = self.answers.get(timeout=.1)
                if message['kind'] == 'error':
                    raise RuntimeError(message['error'])
                return message
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError('Actor exited without a result')
        raise TimeoutError('Actor readiness timeout')

    def stop(self):
        if self.process.is_alive():
            self.commands.put(dict(op='stop'))
        deadline = time.monotonic() + 3
        while self.process.is_alive() and time.monotonic() < deadline:
            try:
                self.answers.get(timeout=.05)
            except queue.Empty:
                pass
            self.process.join(.01)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(10)
        if self.process.is_alive():
            raise RuntimeError('Owned child survived termination')
        self.commands.close()
        self.answers.close()


def replay(nv, actors, job, args, path):
    block_id = f"{job['case_id']}__{job['policy']}__g{job['logical_gpu']}__r{job['repetition']}"
    results, partial = [None]*len(actors), [[] for _ in actors]
    record = dict(schema=SCHEMA, block_id=block_id, status='preparing', job=job,
                  samples=[], results=results, partial_records=partial)
    try:
        for actor, plan in zip(actors, job['plans']):
            actor.commands.put(dict(op='prepare', block_id=block_id, plan=plan))
        for actor in actors:
            if actor.receive(30) != dict(kind='prepared', block_id=block_id):
                raise RuntimeError('Worker PREPARE mismatch')
        origin = time.monotonic() + 1
        start, end = origin + args.warm_s, origin + args.warm_s + args.duration_s
        deadline = end + args.drain_s
        record.update(status='running', origin=origin, window=dict(start=start, end=end), deadline=deadline)
        for actor in actors:
            actor.commands.put(dict(op='go', block_id=block_id, origin=origin))
        last_checkpoint = time.monotonic()
        while True:
            iteration = time.monotonic()
            record['samples'].append(nv.sample())
            for i, actor in enumerate(actors):
                if results[i] is not None:
                    continue
                while True:
                    try:
                        message = actor.answers.get_nowait()
                    except queue.Empty:
                        break
                    if message.get('block_id') != block_id:
                        raise RuntimeError('Worker block mismatch')
                    kind = message['kind']
                    if kind == 'error':
                        record['child_error'] = message
                        raise RuntimeError(message['error'])
                    if kind == 'progress':
                        partial[i].extend(message['records'])
                    elif kind == 'result':
                        results[i] = message
                    else:
                        raise RuntimeError('Unexpected worker message')
                if results[i] is None and not actor.process.is_alive():
                    raise RuntimeError('Worker died while processing trace')
            now = time.monotonic()
            if all(r is not None for r in results):
                target = max([end] + [max((r['finish'] for r in result['records']), default=end)
                                      for result in results])
                if record['samples'][-1]['t'] >= target:
                    break
            if now >= deadline + 5:
                record['coordinator_timeout'] = True
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = dict(kind='censored', records=partial[i], batches=[], drain_capped=True)
                break
            if now-last_checkpoint > 5:
                gpu.atomic_json(path.with_suffix('.partial.json'), record)
                last_checkpoint = now
            time.sleep(max(0, args.sample_s - (time.monotonic()-iteration)))
        last_finish = max([end] + [max((r['finish'] for r in result['records']), default=end)
                                  for result in results])
        drain_end = min(last_finish, record['samples'][-1]['t'])
        record['energy_window'] = telemetry_summary(record['samples'], start, end, job['frequency_MHz'], job['power_cap_W'])
        record['energy_through_drain'] = telemetry_summary(record['samples'], start, drain_end, job['frequency_MHz'], job['power_cap_W'])
        record['drain_end'] = drain_end
        record['replica_summaries'] = []
        for replica, plan, result in zip(job['replicas'], job['plans'], results):
            summary = timing_summary(result['records'], plan['requests'], replica['slo_s'])
            summary.update(replica_id=replica['replica_id'], service_id=replica['service_id'],
                           completed_utility=summary['completed']*replica['utility_per_request'])
            record['replica_summaries'].append(summary)
        record['status'] = 'complete_censored' if any(r['drain_capped'] for r in results) else 'complete'
        if record['energy_window']['energy_J'] is None or record['energy_through_drain']['energy_J'] is None:
            raise RuntimeError('Telemetry does not bracket the complete energy interval')
    except BaseException:
        record.update(status='error', error=traceback.format_exc())
        raise
    finally:
        record['finished'] = time.monotonic()
        gpu.atomic_json(path, record)
    return record


def next_attempt(directory):
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while (directory/f'attempt_{number:03d}').exists():
        number += 1
    path = directory/f'attempt_{number:03d}'
    path.mkdir()
    return path


def execution_identity(job):
    """Reuse identical work across policies or SLO cells of the same trace.

    Replica IDs are labels. Actual configurations, arrivals, image IDs and rates
    must match, so materially different routing cannot be reused accidentally.
    """
    replicas = []
    for replica, plan in zip(job['replicas'], job['plans']):
        replicas.append(dict(
            service=replica['service_id'], arch=replica['arch'], B=replica['B'],
            slices=replica['slices'], rate_rps=replica['rate_rps'],
            utility_per_request=replica['utility_per_request'],
            caller_shape=replica['caller_shape'], requests=plan['requests']))
    return digest(dict(trace_id=job['trace_id'], repetition=job['repetition'],
                       logical_gpu=job['logical_gpu'], mode=job['mode'],
                       frequency_MHz=job['frequency_MHz'], power_cap_W=job['power_cap_W'],
                       layout_slices=sorted(job['layout_slices']),
                       replicas=replicas))


def verified_manifest(marker, root):
    manifest = json.loads(marker.read_text())
    if 'equivalent_to' in manifest:
        source = root/manifest['equivalent_to']/'COMPLETE'
        if file_hash(source) != manifest['source_manifest_sha256']:
            raise RuntimeError('Equivalent source manifest changed')
        source_manifest = json.loads(source.read_text())
        if 'equivalent_to' in source_manifest:
            raise RuntimeError('Nested reuse aliases are not permitted')
        if source_manifest['execution_identity'] != manifest['execution_identity']:
            raise RuntimeError('Reused observation is not execution-equivalent')
        marker, manifest = source, source_manifest
    for filename, expected in manifest['files'].items():
        if file_hash(marker.parent/manifest['attempt']/filename) != expected:
            raise RuntimeError('Archived output hash mismatch: '+filename)
    return marker, manifest


def cleanup_layout(gpu_index, owned):
    # Driver clients may take a moment to release a CUDA context after child exit.
    for attempt in range(6):
        try:
            gpu.destroy_layout(gpu_index, owned)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1)


def run_job(args, nv, job, gpu_uuid):
    name = f"{job['case_id']}__{job['policy']}__g{job['logical_gpu']}__r{job['repetition']}"
    root = args.outdir/'blocks'/name
    fingerprint = digest(dict(job=job, protocol=args.protocol_hash))
    execution_key = execution_identity(job)
    complete = root/'COMPLETE'
    if complete.exists():
        old = json.loads(complete.read_text())
        if old['fingerprint'] != fingerprint or old['gpu_uuid'] != gpu_uuid:
            raise RuntimeError('Resume identity mismatch for ' + name)
        verified_manifest(complete, args.outdir)
        print('SKIP', name, flush=True)
        return
    if execution_key in args.reuse_index:
        source_marker = args.reuse_index[execution_key]
        source = json.loads(source_marker.read_text())
        # Do not reuse censored or failed work. A missing valid observation is rerun.
        if source['status'] == 'complete' and source['gpu_uuid'] == gpu_uuid:
            for filename, expected in source['files'].items():
                if file_hash(source_marker.parent/source['attempt']/filename) != expected:
                    raise RuntimeError('Equivalent source block failed its hash check')
            root.mkdir(parents=True, exist_ok=True)
            gpu.atomic_json(complete, dict(fingerprint=fingerprint, gpu_uuid=gpu_uuid,
                status='complete', execution_identity=execution_key,
                equivalent_to=str(source_marker.parent.relative_to(args.outdir)), job=job,
                source_manifest_sha256=file_hash(source_marker)))
            print('REUSE', name, 'from', source_marker.parent.name, flush=True)
            return
    attempt = next_attempt(root)
    actors, owned, error, cleanup_errors = [], None, None, []
    ownership = attempt/'ownership.json'
    gpu.atomic_json(attempt/'identity.json', dict(schema=SCHEMA, fingerprint=fingerprint, job=job,
                    gpu_uuid=gpu_uuid, gpu=args.gpu, hostname=socket.gethostname(), started=time.time()))
    try:
        gpu.assert_empty(args.gpu)
        gpu.smi('-i', args.gpu, '-pl', job['power_cap_W'])
        actual_cap = float(gpu.smi('-i', args.gpu, '--query-gpu=power.limit', '--format=csv,noheader,nounits'))
        if abs(actual_cap-job['power_cap_W']) > .05:
            raise RuntimeError('Driver did not apply requested power cap')
        if job['mode'] == 'idle_unlocked':
            gpu.smi('-i', args.gpu, '-rgc')
        else:
            gpu.smi('-i', args.gpu, '-lgc', f"{job['frequency_MHz']},{job['frequency_MHz']}")
        assigned, owned = gpu.create_layout(args.gpu, job['layout_slices'], ownership)
        gpu.atomic_json(attempt/'layout.json', dict(assigned=assigned, state=owned))
        available = list(assigned)
        for replica in job['replicas']:
            slot = next(x for x in available if x['slices'] == replica['slices'])
            available.remove(slot)
            actors.append(Actor(slot['uuid'], replica, args))
        metadata = [actor.receive(args.startup_timeout) for actor in actors]
        if any(meta['kind'] != 'ready' for meta in metadata):
            raise RuntimeError('Missing worker READY')
        gpu.atomic_json(attempt/'actors.json', metadata)
        time.sleep(args.settle_s)
        record = replay(nv, actors, job, args, attempt/'replay.json')
    except BaseException:
        error = traceback.format_exc()
        cancelled = not isinstance(sys.exc_info()[1], Exception)
    else:
        cancelled = False
    finally:
        for actor in actors:
            try:
                actor.stop()
            except Exception:
                cleanup_errors.append(traceback.format_exc())
        if owned is not None:
            try:
                cleanup_layout(args.gpu, owned)
            except Exception:
                cleanup_errors.append(traceback.format_exc())
        elif ownership.exists():
            cleanup_errors.append('MIG creation incomplete; audit ownership before continuing')
        gpu.atomic_json(attempt/'outcome.json', dict(error=error, cleanup_errors=cleanup_errors, finished=time.time()))
    if cleanup_errors:
        raise RuntimeError('Cleanup failed; GPU modifications stopped: ' + str(cleanup_errors))
    if cancelled:
        raise KeyboardInterrupt('Replay cancelled after cleanup')
    if error:
        print('ERROR', name, error.splitlines()[-1], flush=True)
        return
    files = {p.name: file_hash(p) for p in attempt.iterdir() if p.is_file()}
    gpu.atomic_json(complete, dict(fingerprint=fingerprint, gpu_uuid=gpu_uuid,
                                 attempt=attempt.name, status=record['status'], files=files,
                                 execution_identity=execution_key))
    if record['status'] == 'complete':
        args.reuse_index[execution_key] = complete
    print('COMPLETE', name, record['status'], round(record['energy_window']['mean_power_W'], 3), flush=True)


def write_immutable(path, value):
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise RuntimeError('Immutable artifact differs: ' + str(path))
    else:
        gpu.atomic_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--outdir', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--torch-home')
    parser.add_argument('--images')
    parser.add_argument('--duration-s', type=float, default=30)
    parser.add_argument('--warm-s', type=float, default=5)
    parser.add_argument('--drain-s', type=float, default=120)
    parser.add_argument('--settle-s', type=float, default=3)
    parser.add_argument('--warmup-batches', type=int, default=20)
    parser.add_argument('--sample-s', type=float, default=.02)
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--max-blocks', type=int)
    parser.add_argument('--case')
    parser.add_argument('--policy')
    args = parser.parse_args()
    if args.repetitions < 1 or args.lanes < 1 or not 0 <= args.lane < args.lanes:
        parser.error('Invalid repetitions or lane assignment')
    times = (args.duration_s, args.warm_s, args.drain_s, args.sample_s,
             args.settle_s, args.startup_timeout)
    if (not all(math.isfinite(t) for t in times) or args.duration_s <= 0
            or args.warm_s < 0 or args.drain_s <= 0 or args.sample_s <= 0
            or args.settle_s < 0 or args.startup_timeout <= 0
            or args.warmup_batches < 0):
        parser.error('Invalid time bounds')
    if args.max_blocks is not None and args.max_blocks < 1:
        parser.error('--max-blocks must be positive')
    cases = normalize_selection(json.loads(args.selection.read_text()))
    jobs, traces = prepare_jobs(cases, args.repetitions, args.warm_s, args.duration_s)
    source_hashes = {name: file_hash(Path(__file__).with_name(name)) for name in ('replay.py', 'gpu_support.py')}
    protocol = dict(schema=SCHEMA, selection_sha256=file_hash(args.selection), source_sha256=source_hashes,
                    duration_s=args.duration_s, warm_s=args.warm_s, drain_s=args.drain_s,
                    settle_s=args.settle_s, warmup_batches=args.warmup_batches, sample_s=args.sample_s,
                    repetitions=args.repetitions, lanes=args.lanes, weights=WEIGHTS,
                    lane_assignment='(logical_gpu + repetition + group_index) modulo lanes',
                    lane_group_definition='Lexicographically sorted unique (trace_id, power_cap_W) groups in the full manifest, before launch filtering; independent of scheduler and SLO',
                    lane_groups=[dict(group_index=index, trace_id=trace_id, power_cap_W=cap)
                        for index, trace_id, cap in sorted({(j['lane_group_index'], j['trace_id'], j['power_cap_W']) for j in jobs})],
                    routing='policy-independent seeded categorical draws; weights are per-replica rates',
                    input='first eight archived real RGB images, per-request IDs shared across policies',
                    partial_batch='pad at arrival-window end, mark padding and retain all real requests',
                    fleet='independent logical boards replayed sequentially, not a simultaneous end-to-end cluster',
                    empty_gpu='measure each explicitly listed unused GPU at its configured idle mode')
    args.protocol_hash = digest(protocol)
    args.reuse_index = {}
    if args.outdir and (args.outdir/'blocks').exists():
        for marker in sorted((args.outdir/'blocks').glob('*/COMPLETE')):
            old = json.loads(marker.read_text())
            if old.get('status') == 'complete' and 'equivalent_to' not in old and old.get('execution_identity'):
                args.reuse_index[old['execution_identity']] = marker
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        write_immutable(args.outdir/'protocol.json', protocol)
        write_immutable(args.outdir/'traces.json', traces)
        write_immutable(args.outdir/'prepared_jobs.json', jobs)
    schedule = [j for j in jobs if lane_for_job(j, args.lanes) == args.lane]
    if args.case:
        schedule = [j for j in schedule if j['case_id'] == args.case]
    if args.policy:
        schedule = [j for j in schedule if j['policy'] == args.policy]
    if args.max_blocks:
        schedule = schedule[:args.max_blocks]
    print(json.dumps(dict(protocol_sha256=args.protocol_hash, cases=len(cases),
                          planned_blocks=len(jobs), selected_blocks=len(schedule),
                          lane=args.lane, lanes=args.lanes)), flush=True)
    if not args.run:
        return
    if any(value is None for value in (args.gpu, args.outdir, args.images, args.torch_home)):
        parser.error('--run requires --gpu --outdir --images --torch-home')
    if os.geteuid() == 0:
        parser.error('Run Python as an ordinary user; only nvidia-smi is elevated')
    if not schedule:
        return
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    import fcntl
    with open(f'/tmp/efficient_mig_gpu{args.gpu}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu.assert_empty(args.gpu)
        initial = gpu.smi('-i', args.gpu, '-q')
        if not re.search(r'MIG Mode\s+Current\s+: Enabled', initial):
            raise RuntimeError('MIG must already be enabled; this runner never changes MIG mode')
        uuid = gpu.smi('-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader')
        original_cap = float(gpu.smi('-i', args.gpu, '--query-gpu=power.limit', '--format=csv,noheader,nounits'))
        gpu.atomic_json(args.outdir/'launches'/f'{time.time_ns()}.json', dict(
            hostname=socket.gethostname(), pid=os.getpid(), gpu=args.gpu, gpu_uuid=uuid,
            lane=args.lane, protocol_hash=args.protocol_hash, original_cap_W=original_cap,
            initial_nvidia_smi=initial, started=time.time()))
        nv = gpu.NVML(args.gpu)
        try:
            for job in schedule:
                run_job(args, nv, job, uuid)
        finally:
            nv.close()
            restoration = {}
            try:
                gpu.assert_empty(args.gpu)
            except Exception:
                restoration['blocked'] = 'Device is no longer empty; leave modes untouched for ownership audit'
                restoration['error'] = traceback.format_exc()
            else:
                for key, command in [('clocks', ('-rgc',)), ('power_cap', ('-pl', original_cap))]:
                    try:
                        restoration[key] = gpu.smi('-i', args.gpu, *command)
                    except Exception:
                        restoration[key+'_error'] = traceback.format_exc()
            restoration['final_nvidia_smi'] = gpu.smi('-i', args.gpu, '-q')
            gpu.atomic_json(args.outdir/f'restoration_{time.time_ns()}.json', restoration)


if __name__ == '__main__':
    main()
